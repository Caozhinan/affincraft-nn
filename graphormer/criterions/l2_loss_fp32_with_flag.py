from fairseq.dataclass.configs import FairseqDataclass  
import torch  
import torch.nn as nn  
from fairseq import metrics  
from fairseq.criterions import FairseqCriterion, register_criterion 


@register_criterion("l2_loss_fp32_with_flag", dataclass=FairseqDataclass)  
class GraphPredictionL2LossFP32WithFlag(FairseqCriterion):  
    """FP32 训练专用的 L2 Loss with FLAG 机制 (静态数据 RMSD=0)"""  
      
    def forward(self, model, sample, reduce=True):  
        model_device = next(model.parameters()).device  
        sample = self._move_to_device(sample, model_device)  
          
        if sample["net_input"]["batched_data"] is None:  
            return self._zero_loss(model_device)  
          
        sample_size = sample["nsamples"]  
          
        with torch.no_grad():  
            natoms = sample["net_input"]["batched_data"]["node_feat"].shape[1]  
          
        perturb = sample.get("perturb", None)  
          
        try:  
            logits = model(**sample["net_input"], perturb=perturb)  
              
            if isinstance(logits, tuple):  
                logits, weights = logits  
            else:  
                weights = torch.ones(logits.shape, dtype=logits.dtype, device=logits.device)  
              
            targets = model.get_targets(sample, [logits])  
              
            # 使用静态数据的归一化参数  
            targets_normalize = (targets - 7.044721989436028) / 1.352596540727069  
              
            standard_loss = nn.MSELoss(reduction="none")(  
                logits, targets_normalize[:logits.size(0)]  
            )  
              
            loss = (standard_loss * weights).sum()  
              
            if not torch.isfinite(loss):  
                print(f"[CRITERION FP32 FLAG] Non-finite loss: {loss.item()}", flush=True)  
                return self._zero_loss(model_device)  
              
            logging_output = {  
                "loss": loss.data,  
                "sample_size": logits.size(0),  
                "nsentences": sample_size,  
                "ntokens": natoms,  
            }  
              
            return loss, sample_size, logging_output  
              
        except Exception as e:  
            print(f"[CRITERION FP32 FLAG] Exception: {str(e)}", flush=True)  
            return self._zero_loss(model_device)  
      
    def _move_to_device(self, obj, device):  
        if isinstance(obj, torch.Tensor):  
            return obj.to(device)  
        elif isinstance(obj, dict):  
            return {k: self._move_to_device(v, device) for k, v in obj.items()}  
        elif isinstance(obj, list):  
            return [self._move_to_device(v, device) for v in obj]  
        else:  
            return obj  
      
    def _zero_loss(self, device):  
        return torch.tensor(0.0, device=device, requires_grad=False), 0, {  
            "loss": 0.0, "sample_size": 0, "nsentences": 0, "ntokens": 0  
        }  
      
    @staticmethod  
    def reduce_metrics(logging_outputs) -> None:  
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)  
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)  
        if sample_size > 0:  
            metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)  
      
    @staticmethod  
    def logging_outputs_can_be_summed() -> bool:  
        return True

@register_criterion("l2_loss_fp32", dataclass=FairseqDataclass)  
class GraphPredictionL2LossFP32(FairseqCriterion):  
    """FP32 训练专用的 L2 Loss - 带 NaN 检测和分布式同步"""  
      
    def forward(self, model, sample, reduce=True):  
        model_device = next(model.parameters()).device  
        sample = self._move_to_device(sample, model_device)  
          
        if sample["net_input"]["batched_data"] is None:  
            return self._zero_loss(model_device)  
          
        sample_size = sample["nsamples"]  
        sample_id = sample.get("pdbid", "unknown")  
          
        # 检查输入数据中的 NaN  
        nan_locations = []  
        should_skip = False  
          
        for key, value in sample["net_input"].items():  
            if isinstance(value, dict):  
                for subkey, subvalue in value.items():  
                    if self._check_tensor_for_nan(subvalue, f"input[{key}][{subkey}]", sample_id):  
                        nan_locations.append(f"input[{key}][{subkey}]")  
                        should_skip = True  
            else:  
                if self._check_tensor_for_nan(value, f"input[{key}]", sample_id):  
                    nan_locations.append(f"input[{key}]")  
                    should_skip = True  
          
        # 分布式同步：确保所有 rank 一致跳过  
        if torch.distributed.is_initialized():  
            skip_tensor = torch.tensor(1 if should_skip else 0, device=model_device, dtype=torch.long)  
            torch.distributed.all_reduce(skip_tensor, op=torch.distributed.ReduceOp.MAX)  
            should_skip = skip_tensor.item() > 0  
          
        if should_skip:  
            print(f"\n{'='*80}")  
            print(f"[CRITERION] NaN DETECTED - Skipping batch")  
            print(f"[CRITERION] Sample ID: {sample_id}")  
            print(f"[CRITERION] NaN locations: {', '.join(nan_locations)}")  
            print(f"{'='*80}\n")  
            return self._zero_loss(model_device)  
          
        # 继续正常的前向传播  
        try:  
            logits = model(**sample["net_input"])  
              
            if isinstance(logits, tuple):  
                logits, weights = logits  
            else:  
                weights = torch.ones(logits.shape, dtype=logits.dtype, device=logits.device)  
              
            # 检查模型输出  
            if self._check_tensor_for_nan(logits, "model output", sample_id):  
                print(f"[CRITERION] NaN in model output - Skipping batch")  
                return self._zero_loss(model_device)  
              
            targets = model.get_targets(sample, [logits])  
            targets_normalize = (targets - 7.044721989436028) / 1.352596540727069  
              
            loss = nn.MSELoss(reduction="none")(logits, targets_normalize[:logits.size(0)])  
            loss = (loss * weights).sum()  
              
            if not torch.isfinite(loss):  
                print(f"[CRITERION] Non-finite loss detected: {loss.item()}")  
                return self._zero_loss(model_device)  
              
            logging_output = {  
                "loss": loss.data,  
                "sample_size": logits.size(0),  
                "nsentences": sample_size,  
                "ntokens": sample["net_input"]["batched_data"]["node_feat"].shape[1],  
            }  
              
            return loss, sample_size, logging_output  
              
        except Exception as e:  
            print(f"[CRITERION] Exception: {str(e)}")  
            return self._zero_loss(model_device)  
      
    def _check_tensor_for_nan(self, tensor, name, sample_id=None):  
        """检查张量中的 NaN/Inf 并打印详细信息"""  
        if not isinstance(tensor, torch.Tensor):  
            return False  
          
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():  
            sample_info = f" in sample {sample_id}" if sample_id else ""  
            print(f"\n[NaN DETECTED] {name}{sample_info}")  
            print(f"  Shape: {tensor.shape}")  
            print(f"  Dtype: {tensor.dtype}")  
            print(f"  Device: {tensor.device}")  
              
            with torch.no_grad():  
                nan_count = torch.isnan(tensor).sum().item()  
                inf_count = torch.isinf(tensor).sum().item()  
                print(f"  NaN count: {nan_count}, Inf count: {inf_count}")  
                  
                finite_mask = torch.isfinite(tensor)  
                if finite_mask.any():  
                    finite_vals = tensor[finite_mask]  
                    print(f"  Finite range: [{finite_vals.min().item():.4f}, {finite_vals.max().item():.4f}]")  
            return True  
        return False  
      
    def _zero_loss(self, device):  
        """返回零损失 - 保持梯度流"""  
        return torch.tensor(0.0, device=device, requires_grad=True), 0, {  
            "loss": 0.0,  
            "sample_size": 0,  
            "nsentences": 0,  
            "ntokens": 0  
        }  
      
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