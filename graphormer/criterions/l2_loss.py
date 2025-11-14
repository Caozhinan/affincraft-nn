# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from fairseq.dataclass.configs import FairseqDataclass
import torch
import torch.nn as nn
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion

# 在现有的 import 语句之后添加  
import logging  
  
# 启用 PyTorch 异常检测  
# torch.autograd.set_detect_anomaly(True)  
  
# 创建 logger 实例  
logger = logging.getLogger(__name__)  
  
# 添加 NaN/Inf 检测辅助函数  
def check_tensor_for_nan(tensor, name, sample_id=None):  
    """优化版本:一次性检测 NaN/Inf,避免多次调用"""  
    if not isinstance(tensor, torch.Tensor):  
        return False  
      
    # 使用 torch.isfinite 一次性检测 NaN 和 Inf  
    is_finite = torch.isfinite(tensor)  
    has_issue = not is_finite.all()  
      
    if has_issue:  
        sample_info = f" in sample {sample_id}" if sample_id is not None else ""  
        # 只在需要时计算统计信息  
        with torch.no_grad():  
            nan_count = (~is_finite).sum().item()  
            finite_mask = is_finite  
            if finite_mask.any():  
                finite_vals = tensor[finite_mask]  
                min_val = finite_vals.min().item()  
                max_val = finite_vals.max().item()  
                mean_val = finite_vals.mean().item()  
            else:  
                min_val = max_val = mean_val = float('nan')  
          
        print("=" * 80)  
        print(f"[NaN/Inf DETECTED] {name}{sample_info}")  
        print(f"  Tensor shape: {tensor.shape}")  
        print(f"  Non-finite count: {nan_count}")  
        print(f"  Finite values - Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {mean_val:.4f}")  
        print("=" * 80)  
        return True  
      
    return False


@register_criterion("l2_loss_rmsd_fp32", dataclass=FairseqDataclass)  
class GraphPredictionL2LossWithRMSD_FP32(FairseqCriterion):  
    """  
    FP32 训练专用版本的 L2 Loss with RMSD-based refinement.  
    移除了所有 FP16 防护措施和分布式同步,避免训练挂起问题.  
    """  
      
    def forward(self, model, sample, reduce=True):  
        """Compute the loss for the given sample with RMSD-based refinement."""  
        # 获取模型设备  
        model_device = next(model.parameters()).device  
          
        # 将 sample 移动到模型设备  
        sample = self._move_to_device(sample, model_device)  
          
        # 检查 batched_data 是否为 None  
        if sample["net_input"]["batched_data"] is None:  
            return self._zero_loss(model_device)  
          
        sample_size = sample["nsamples"]  
          
        # 获取原子数量  
        with torch.no_grad():  
            natoms = sample["net_input"]["batched_data"]["node_feat"].shape[1]  
          
        try:  
            # 前向传播  
            logits = model(**sample["net_input"])  
              
            # 处理权重  
            if isinstance(logits, tuple):  
                logits, weights = logits  
            else:  
                weights = torch.ones(logits.shape, dtype=logits.dtype, device=logits.device)  
              
            # 获取目标值  
            targets = model.get_targets(sample, [logits])  
              
            # 标准化目标值 (使用 MD 数据的均值和标准差)  
            targets_normalize = (targets - 6.5227203013597315) / 1.8651215830061156  
              
            # 计算标准 MSE loss  
            standard_loss = nn.MSELoss(reduction="none")(  
                logits, targets_normalize[:logits.size(0)]  
            )  
              
            # RMSD 加权逻辑  
            rmsd_values = sample["net_input"]["batched_data"].get("rmsd", None)  
              
            if rmsd_values is None:  
                # 没有 RMSD 值,使用标准 loss  
                loss = standard_loss  
            else:  
                # 有 RMSD 值,应用平滑加权  
                prediction_error = logits - targets_normalize[:logits.size(0)]  
                  
                # RMSD 加权参数  
                rmsd_threshold = 2.0  
                rmsd_steepness = 5.0  
                error_steepness = 10.0  
                min_weight = 0.1  
                  
                # 计算 RMSD 权重 (RMSD 越小权重越大)  
                rmsd_weight = torch.sigmoid(rmsd_steepness * (rmsd_threshold - rmsd_values))  
                  
                # 计算预测误差惩罚  
                error_penalty = torch.sigmoid(error_steepness * prediction_error.squeeze(-1))  
                error_weight = min_weight + (1 - min_weight) * error_penalty  
                  
                # 组合权重  
                combined_weight = (rmsd_weight + (1 - rmsd_weight) * error_weight).unsqueeze(-1)  
                  
                # 应用权重  
                loss = combined_weight * standard_loss  
              
            # 最终 loss 计算  
            loss = (loss * weights).sum()  
              
            # 简单的有效性检查 (不做分布式同步)  
            if not torch.isfinite(loss):  
                print(f"[CRITERION FP32] Non-finite loss detected: {loss.item()}")  
                return self._zero_loss(model_device)  
              
            # 构建 logging output  
            logging_output = {  
                "loss": loss.data,  
                "sample_size": logits.size(0),  
                "nsentences": sample_size,  
                "ntokens": natoms,  
            }  
              
            return loss, sample_size, logging_output  
              
        except Exception as e:  
            # 异常处理 - 简化版,只打印错误信息  
            print(f"[CRITERION FP32] Exception during forward pass: {str(e)}")  
            return self._zero_loss(model_device)  
      
    def _move_to_device(self, obj, device):  
        """递归地将对象移动到指定设备"""  
        if isinstance(obj, torch.Tensor):  
            return obj.to(device)  
        elif isinstance(obj, dict):  
            return {k: self._move_to_device(v, device) for k, v in obj.items()}  
        elif isinstance(obj, list):  
            return [self._move_to_device(v, device) for v in obj]  
        else:  
            return obj  
      
    def _zero_loss(self, device):  
        """返回零损失"""  
        return torch.tensor(0.0, device=device, requires_grad=False), 0, {  
            "loss": 0.0,  
            "sample_size": 0,  
            "nsentences": 0,  
            "ntokens": 0  
        }  
      
    @staticmethod  
    def reduce_metrics(logging_outputs) -> None:  
        """Aggregate logging outputs from data parallel training."""  
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)  
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)  
          
        if sample_size > 0:  
            metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)  
      
    @staticmethod  
    def logging_outputs_can_be_summed() -> bool:  
        """  
        Whether the logging outputs returned by `forward` can be summed  
        across workers prior to calling `reduce_metrics`. Setting this  
        to True will improves distributed training speed.  
        """  
        return True

@register_criterion("l2_loss_rmsd", dataclass=FairseqDataclass)  
class GraphPredictionL2LossWithRMSD(FairseqCriterion):  
    """  
    Implementation for the L2 loss with RMSD-based refinement used in graphormer model training.  
    包含优化的 NaN 检测、loss 钳制、梯度钳制和分布式同步功能。  
    """  
      
    def forward(self, model, sample, reduce=True):  
        """Compute the loss for the given sample with RMSD-based refinement."""  
        model_device = next(model.parameters()).device  
          
        # 将整个 sample 移动到模型设备  
        def move_to_device(obj, device):  
            if isinstance(obj, torch.Tensor):  
                return obj.to(device)  
            elif isinstance(obj, dict):  
                return {k: move_to_device(v, device) for k, v in obj.items()}  
            elif isinstance(obj, list):  
                return [move_to_device(v, device) for v in obj]  
            else:  
                return obj  
    
        sample = move_to_device(sample, model_device)  
          
        # 检查 batched_data 是否为 None  
        if sample["net_input"]["batched_data"] is None:  
            return torch.tensor(0.0, device=model_device, requires_grad=False), 0, {  
                "loss": 0.0, "sample_size": 0, "nsentences": 0, "ntokens": 0  
            }  
          
        sample_size = sample["nsamples"]  
        sample_id = sample.get("pdbid", None)  
          
        # 添加分布式同步标志  
        should_skip = False  
        nan_locations = []  
          
        # Stage 1: 检查输入数据  
        for key, value in sample["net_input"].items():  
            if isinstance(value, dict):  
                for subkey, subvalue in value.items():  
                    if check_tensor_for_nan(subvalue, f"input[{key}][{subkey}]", sample_id):  
                        nan_locations.append(f"input[{key}][{subkey}]")  
                        should_skip = True  
            else:  
                if check_tensor_for_nan(value, f"input[{key}]", sample_id):  
                    nan_locations.append(f"input[{key}]")  
                    should_skip = True  
          
        # 分布式同步跳过标志  
        if torch.distributed.is_initialized():  
            skip_tensor = torch.tensor(1 if should_skip else 0, device=model_device, dtype=torch.long)  
            torch.distributed.all_reduce(skip_tensor, op=torch.distributed.ReduceOp.MAX)  
            should_skip = skip_tensor.item() > 0  
          
        if should_skip:  
            print(f"\n{'='*80}")  
            print(f"[CRITERION NaN DETECTED] Sample ID: {sample_id}")  
            print(f"[CRITERION] NaN locations: {', '.join(nan_locations)}")  
            print(f"{'='*80}\n")  
              
            return torch.tensor(0.0, device=model_device, requires_grad=False), 0, {  
                "loss": 0.0, "sample_size": 0, "nsentences": 0, "ntokens": 0  
            }  
          
        with torch.no_grad():  
            natoms = sample["net_input"]["batched_data"]["node_feat"].shape[1]  
          
        try:  
            # Stage 2: 模型前向传播  
            logits = model(**sample["net_input"])  
            if isinstance(logits, tuple):  
                logits, weights = logits  
                if check_tensor_for_nan(weights, "model weights", sample_id):  
                    nan_locations.append("model weights")  
                    should_skip = True  
            else:  
                weights = torch.ones(logits.shape, dtype=logits.dtype, device=logits.device)  
              
            if check_tensor_for_nan(logits, "model output (logits)", sample_id):  
                nan_locations.append("model output (logits)")  
                should_skip = True  
              
            # Stage 3: 检查目标值  
            targets = model.get_targets(sample, [logits])  
            if check_tensor_for_nan(targets, "targets", sample_id):  
                nan_locations.append("targets")  
                should_skip = True  
              
            # 分布式同步 - 模型输出阶段  
            if torch.distributed.is_initialized():  
                skip_tensor = torch.tensor(1 if should_skip else 0, device=logits.device, dtype=torch.long)  
                torch.distributed.all_reduce(skip_tensor, op=torch.distributed.ReduceOp.MAX)  
                should_skip = skip_tensor.item() > 0  
              
            if should_skip:  
                print(f"\n{'='*80}")  
                print(f"[CRITERION FORWARD NaN] Sample ID: {sample_id}")  
                print(f"[CRITERION] NaN in forward pass: {', '.join(nan_locations)}")  
                print(f"{'='*80}\n")  
                  
                return torch.tensor(0.0, device=logits.device, requires_grad=False), 0, {  
                    "loss": 0.0, "sample_size": 0, "nsentences": 0, "ntokens": 0  
                }  
              
            # Stage 4-6: 继续原有的loss计算逻辑  
            targets_normalize = (targets - 6.5227203013597315) / 1.8651215830061156  
              
            rmsd_values = sample["net_input"]["batched_data"].get("rmsd", None)  
            if rmsd_values is None:  
                standard_loss = nn.MSELoss(reduction="none")(logits, targets_normalize[: logits.size(0)])  
                loss = standard_loss  
            else:  
                standard_loss = nn.MSELoss(reduction="none")(logits, targets_normalize[: logits.size(0)])  
                prediction_error = logits - targets_normalize[: logits.size(0)]  
                  
                rmsd_threshold = 2.0  
                rmsd_steepness = 5.0  
                error_steepness = 10.0  
                min_weight = 0.1  
                  
                rmsd_weight = torch.sigmoid(rmsd_steepness * (rmsd_threshold - rmsd_values))  
                error_penalty = torch.sigmoid(error_steepness * prediction_error.squeeze(-1))  
                error_weight = min_weight + (1 - min_weight) * error_penalty  
                combined_weight = (rmsd_weight + (1 - rmsd_weight) * error_weight).unsqueeze(-1)  
                loss = combined_weight * standard_loss  
              
            # Stage 7: 最终损失计算  
            loss = (loss * weights).sum()  
            loss = torch.clamp(loss, min=-1e4, max=1e4)  
              
            # 最终检查  
            if not torch.isfinite(loss):  
                print(f"\n{'='*80}")  
                print(f"[CRITERION LOSS NaN] Sample ID: {sample_id}")  
                print(f"[CRITERION] Non-finite loss before backward!")  
                print(f"{'='*80}\n")  
                  
                return torch.tensor(0.0, device=logits.device, requires_grad=False), 0, {  
                    "loss": 0.0, "sample_size": 0, "nsentences": 0, "ntokens": 0  
                }  
              
            # 分布式同步 - 最终 loss 有效性检查  
            if torch.distributed.is_initialized():  
                loss_valid = torch.isfinite(loss)  
                loss_valid_tensor = torch.tensor(1 if loss_valid else 0, device=loss.device, dtype=torch.long)  
                torch.distributed.all_reduce(loss_valid_tensor, op=torch.distributed.ReduceOp.MIN)  
                  
                if loss_valid_tensor.item() == 0:  
                    print(f"[CRITERION DISTRIBUTED] Sample ID: {sample_id} - Loss invalid on some rank")  
                    return torch.tensor(0.0, device=logits.device, requires_grad=False), 0, {  
                        "loss": 0.0, "sample_size": 0, "nsentences": 0, "ntokens": 0  
                    }  
              
            # 梯度钳制 hook  
            def gradient_clamp_hook(grad):  
                if grad is not None:  
                    grad_finite = torch.isfinite(grad)  
                    if not grad_finite.all():  
                        grad = torch.where(grad_finite, grad, torch.zeros_like(grad))  
                    grad = torch.clamp(grad, min=-1e4, max=1e4)  
                return grad  
              
            if loss.requires_grad:  
                loss.register_hook(gradient_clamp_hook)  
              
            logging_output = {  
                "loss": loss.data,  
                "sample_size": logits.size(0),  
                "nsentences": sample_size,  
                "ntokens": natoms,  
            }  
            return loss, sample_size, logging_output  
          
        # ===== 新增: OOM 专门处理 =====  
        except RuntimeError as e:  
            if "out of memory" in str(e).lower():  
                print(f"\n{'='*80}")  
                print(f"[CRITERION OOM] Sample ID: {sample_id}")  
                print(f"[CRITERION] OOM Error: {str(e)}")  
                print(f"{'='*80}\n")  
                  
                # 分布式 OOM 同步  
                if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:  
                    try:  
                        oom_tensor = torch.tensor(1, device=model_device, dtype=torch.long)  
                        torch.distributed.all_reduce(  
                            oom_tensor,  
                            op=torch.distributed.ReduceOp.MAX  
                        )  
                        print(f"[CRITERION OOM SYNC] Rank {torch.distributed.get_rank()}: "  
                              f"OOM synchronized across all ranks")  
                    except Exception as sync_error:  
                        print(f"[CRITERION OOM SYNC ERROR] Failed to sync OOM: {sync_error}")  
                  
                # 清理显存  
                if torch.cuda.is_available():  
                    torch.cuda.empty_cache()  
                  
                return torch.tensor(0.0, device=model_device, requires_grad=False), 0, {  
                    "loss": 0.0, "sample_size": 0, "nsentences": 0, "ntokens": 0  
                }  
            else:  
                # 非 OOM 的 RuntimeError,重新抛出  
                raise e  
          
        except Exception as e:  
            print(f"\n{'='*80}")  
            print(f"[CRITERION EXCEPTION] Sample ID: {sample_id}")  
            print(f"[CRITERION] Exception: {str(e)}")  
            print(f"[CRITERION] Exception type: {type(e).__name__}")  
            print(f"[CRITERION] Sample keys: {list(sample.keys())}")  
            if "net_input" in sample:  
                print(f"[CRITERION] net_input keys: {list(sample['net_input'].keys())}")  
            import traceback  
            print(f"[CRITERION] Traceback:\n{traceback.format_exc()}")  
            print(f"{'='*80}\n")  
              
            return torch.tensor(0.0, device=model_device, requires_grad=False), 0, {  
                "loss": 0.0, "sample_size": 0, "nsentences": 0, "ntokens": 0  
            }  
      
    @staticmethod  
    def reduce_metrics(logging_outputs) -> None:  
        """Aggregate logging outputs from data parallel training."""  
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)  
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)  
          
        if sample_size > 0:  
            metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)  
      
    @staticmethod  
    def logging_outputs_can_be_summed() -> bool:  
        """  
        Whether the logging outputs returned by `forward` can be summed  
        across workers prior to calling `reduce_metrics`. Setting this  
        to True will improves distributed training speed.  
        """  
        return True


# =====================================================================================
# 二、带 FLAG（普通版）：l2_loss_with_flag
# =====================================================================================
@register_criterion("l2_loss_with_flag", dataclass=FairseqDataclass)
class GraphPredictionL2LossWithFlag(FairseqCriterion):
    """L2 Loss + FLAG + RMSD 平滑加权"""

    def forward(self, model, sample, reduce=True):
        sample_size = sample["nsamples"]
        batch_data = sample["net_input"]["batched_data"]
        perturb = sample.get("perturb", None)
        natoms = batch_data["node_feat"].shape[1]

        logits = model(**sample["net_input"], perturb=perturb)
        if isinstance(logits, tuple):
            logits, weights = logits
        else:
            weights = torch.ones_like(logits)

        targets = model.get_targets(sample, [logits])
        targets_norm = (targets - 6.529300030461668) / 1.9919705951218716
        standard_loss = nn.MSELoss(reduction="none")(logits, targets_norm[: logits.size(0)])

        rmsd_values = batch_data.get("rmsd", None)
        if rmsd_values is not None:
            rmsd_threshold = 2.0
            rmsd_steepness = 5.0
            error_steepness = 10.0
            min_weight = 0.1
            rmsd_weight = torch.sigmoid(rmsd_steepness * (rmsd_threshold - rmsd_values))
            prediction_error = (logits - targets_norm[: logits.size(0)]).squeeze(-1)
            error_penalty = torch.sigmoid(error_steepness * prediction_error)
            error_weight = min_weight + (1 - min_weight) * error_penalty
            combined_weight = (rmsd_weight + (1 - rmsd_weight) * error_weight).unsqueeze(-1)
            loss = combined_weight * standard_loss
        else:
            loss = standard_loss

        loss = (loss * weights).sum()
        logging_output = {
            "loss": loss.data,
            "sample_size": logits.size(0),
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs):
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        size_sum = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / size_sum, size_sum, round=6)

    @staticmethod
    def logging_outputs_can_be_summed():
        return True


# =====================================================================================
# 三、带 FLAG & RMSD (融合版)：l2_loss_rmsd_with_flag
# =====================================================================================
@register_criterion("l2_loss_rmsd_with_flag", dataclass=FairseqDataclass)
class GraphPredictionL2LossWithRMSDAndFlag(FairseqCriterion):
    """融合 FLAG + 平滑 RMSD 加权 L2 损失"""

    def forward(self, model, sample, reduce=True):
        sample_size = sample["nsamples"]
        batch_data = sample["net_input"]["batched_data"]
        perturb = sample.get("perturb", None)
        natoms = batch_data["node_feat"].shape[1]

        logits = model(**sample["net_input"], perturb=perturb)
        if isinstance(logits, tuple):
            logits, weights = logits
        else:
            weights = torch.ones_like(logits)

        targets = model.get_targets(sample, [logits])
        targets_norm = (targets - 6.5227203013597315) / 1.8651215830061156
        standard_loss = nn.MSELoss(reduction="none")(logits, targets_norm[: logits.size(0)])

        rmsd_values = batch_data.get("rmsd", None)
        if rmsd_values is not None:
            rmsd_threshold = 2.0
            rmsd_steepness = 5.0
            error_steepness = 10.0
            min_weight = 0.1
            rmsd_weight = torch.sigmoid(rmsd_steepness * (rmsd_threshold - rmsd_values))
            prediction_error = (logits - targets_norm[: logits.size(0)]).squeeze(-1)
            error_penalty = torch.sigmoid(error_steepness * prediction_error)
            error_weight = min_weight + (1 - min_weight) * error_penalty
            combined_weight = (rmsd_weight + (1 - rmsd_weight) * error_weight).unsqueeze(-1)
            loss = combined_weight * standard_loss
        else:
            loss = standard_loss

        loss = (loss * weights).sum()
        logging_output = {
            "loss": loss.data,
            "sample_size": logits.size(0),
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs):
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        size_sum = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / size_sum, size_sum, round=6)

    @staticmethod
    def logging_outputs_can_be_summed():
        return True

@register_criterion("l2_loss", dataclass=FairseqDataclass)
class GraphPredictionL1Loss(FairseqCriterion):
    """L2 (MAE) loss for graphormer training."""
    acc_loss, inc = 0, 0

    def forward(self, model, sample, reduce=True):
        sample_size = sample["nsamples"]
        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]["node_feat"].shape[1]

        logits = model(**sample["net_input"])
        if isinstance(logits, tuple):
            logits, weights = logits
        else:
            weights = torch.ones(logits.shape, dtype=logits.dtype, device=logits.device)

        targets = model.get_targets(sample, [logits])
        targets_normalize = (targets - 6.529300030461668) / 1.9919705951218716
        loss = nn.MSELoss(reduction="none")(logits, targets_normalize[: logits.size(0)])
        loss = (loss * weights).sum()

        logging_output = {
            "loss": loss.data,
            "sample_size": logits.size(0),
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True