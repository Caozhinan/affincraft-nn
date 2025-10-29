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
torch.autograd.set_detect_anomaly(True)  
  
# 创建 logger 实例  
logger = logging.getLogger(__name__)  
  
# 添加 NaN/Inf 检测辅助函数  
def check_tensor_for_nan(tensor, name, sample_id=None):  
    """Helper function to check for NaN/Inf values in tensors"""  
    if isinstance(tensor, torch.Tensor):  
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():  
            sample_info = f" in sample {sample_id}" if sample_id is not None else ""  
            logger.warning(f"NaN/Inf detected in {name}{sample_info}")  
            logger.warning(f"  Shape: {tensor.shape}")  
            logger.warning(f"  Min: {tensor.min().item()}, Max: {tensor.max().item()}, Mean: {tensor.mean().item()}")  
            return True  
    return False

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


@register_criterion("l2_loss_rmsd", dataclass=FairseqDataclass)    
class GraphPredictionL2LossWithRMSD(FairseqCriterion):    
    """    
    Implementation for the L2 loss with RMSD-based refinement used in graphormer model training.    
    """    
        
    def forward(self, model, sample, reduce=True):    
        """Compute the loss for the given sample with RMSD-based refinement.    
    
        Returns a tuple with three elements:    
        1) the loss    
        2) the sample size, which is used as the denominator for the gradient    
        3) logging outputs to display while training    
        """    
        if sample["net_input"]["batched_data"] is None:  
            return torch.tensor(0.0), 0, {  
                "loss": 0.0,   
                "sample_size": 0,   
                "nsentences": 0,   
                "ntokens": 0  
            }
        sample_size = sample["nsamples"]  
          
        # 获取 sample ID（如果存在）  
        sample_id = sample.get("pdbid", None)  
        if sample_id is not None:  
            logger.info(f"Processing sample: {sample_id}")  
          
        # Stage 1: 检查输入数据  
        for key, value in sample["net_input"].items():  
            if isinstance(value, dict):  
                for subkey, subvalue in value.items():  
                    if check_tensor_for_nan(subvalue, f"input[{key}][{subkey}]", sample_id):  
                        raise RuntimeError(f"NaN/Inf in input data: {key}.{subkey}")  
            else:  
                if check_tensor_for_nan(value, f"input[{key}]", sample_id):  
                    raise RuntimeError(f"NaN/Inf in input data: {key}")  
    
        with torch.no_grad():    
            natoms = sample["net_input"]["batched_data"]["node_feat"].shape[1]  
    
        try:  
            # Stage 2: 检查模型输出  
            logits = model(**sample["net_input"])    
            if isinstance(logits, tuple):    
                logits, weights = logits  
                if check_tensor_for_nan(weights, "model weights", sample_id):  
                    raise RuntimeError("NaN/Inf in model weights")  
            else:    
                weights = torch.ones(logits.shape, dtype=logits.dtype, device=logits.device)  
              
            if check_tensor_for_nan(logits, "model output (logits)", sample_id):  
                raise RuntimeError("NaN/Inf in model logits")  
                
            # Stage 3: 检查目标值  
            targets = model.get_targets(sample, [logits])  
            if check_tensor_for_nan(targets, "targets", sample_id):  
                raise RuntimeError("NaN/Inf in targets")  
                
            # Stage 4: 检查归一化后的目标值  
            targets_normalize = (targets - 6.5227203013597315) / 1.8651215830061156  
            if check_tensor_for_nan(targets_normalize, "normalized targets", sample_id):  
                raise RuntimeError("NaN/Inf in normalized targets")  
        
            # Get RMSD values from sample    
            rmsd_values = sample["net_input"]["batched_data"].get("rmsd", None)  
            if rmsd_values is not None:  
                if check_tensor_for_nan(rmsd_values, "RMSD values", sample_id):  
                    raise RuntimeError("NaN/Inf in RMSD values")  
              
            if rmsd_values is None:    
                # Fallback to standard L2 loss if no RMSD available  
                standard_loss = nn.MSELoss(reduction="none")(logits, targets_normalize[: logits.size(0)])  
                if check_tensor_for_nan(standard_loss, "standard loss (no RMSD)", sample_id):  
                    raise RuntimeError("NaN/Inf in standard loss")  
                loss = standard_loss  
            else:    
                # Apply RMSD-based loss refinement with smooth weighting  
                standard_loss = nn.MSELoss(reduction="none")(logits, targets_normalize[: logits.size(0)])  
                  
                # Stage 5: 检查标准损失  
                if check_tensor_for_nan(standard_loss, "standard loss (before weighting)", sample_id):  
                    raise RuntimeError("NaN/Inf in standard loss before weighting")  
                  
                # 计算预测误差（预测值 - 真实值）  
                prediction_error = logits - targets_normalize[: logits.size(0)]  
                if check_tensor_for_nan(prediction_error, "prediction error", sample_id):  
                    raise RuntimeError("NaN/Inf in prediction error")  
                  
                # 使用 sigmoid 平滑 RMSD 阈值  
                rmsd_threshold = 2.0  
                rmsd_steepness = 5.0  
                rmsd_weight = torch.sigmoid(rmsd_steepness * (rmsd_threshold - rmsd_values))  
                if check_tensor_for_nan(rmsd_weight, "RMSD weight", sample_id):  
                    raise RuntimeError("NaN/Inf in RMSD weight")  
                  
                # 使用 sigmoid 平滑预测误差的惩罚  
                error_steepness = 10.0  
                min_weight = 0.1  
                error_penalty = torch.sigmoid(error_steepness * prediction_error.squeeze(-1))  
                if check_tensor_for_nan(error_penalty, "error penalty", sample_id):  
                    raise RuntimeError("NaN/Inf in error penalty")  
                  
                error_weight = min_weight + (1.0 - min_weight) * error_penalty  
                if check_tensor_for_nan(error_weight, "error weight", sample_id):  
                    raise RuntimeError("NaN/Inf in error weight")  
                  
                # 组合权重  
                combined_weight = (rmsd_weight + (1.0 - rmsd_weight) * error_weight).unsqueeze(-1)  
                if check_tensor_for_nan(combined_weight, "combined weight", sample_id):  
                    raise RuntimeError("NaN/Inf in combined weight")  
                  
                loss = combined_weight * standard_loss  
                if check_tensor_for_nan(loss, "weighted loss (before sum)", sample_id):  
                    raise RuntimeError("NaN/Inf in weighted loss before sum")  
        
            # Stage 6: 检查最终加权损失  
            loss = (loss * weights).sum()  
            if check_tensor_for_nan(loss, "final loss", sample_id):  
                logger.error(f"NaN in final loss! Sample details:")  
                logger.error(f"  - Sample ID: {sample_id}")  
                logger.error(f"  - Logits min/max/mean: {logits.min().item():.4f}/{logits.max().item():.4f}/{logits.mean().item():.4f}")  
                logger.error(f"  - Targets min/max/mean: {targets.min().item():.4f}/{targets.max().item():.4f}/{targets.mean().item():.4f}")  
                raise RuntimeError("NaN/Inf in final loss")  
        
            logging_output = {    
                "loss": loss.data,    
                "sample_size": logits.size(0),    
                "nsentences": sample_size,    
                "ntokens": natoms,    
            }    
            return loss, sample_size, logging_output  
              
        except Exception as e:  
            logger.error(f"ERROR during forward pass: {str(e)}")  
            logger.error(f"Sample ID: {sample_id}")  
            raise e  
  
    @staticmethod  
    def reduce_metrics(logging_outputs) -> None:  
        """Aggregate logging outputs from data parallel training."""  
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)  
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)  
  
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