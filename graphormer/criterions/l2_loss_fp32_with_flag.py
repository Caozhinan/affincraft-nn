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
    """  
    FP32 训练专用的 L2 Loss (静态数据 RMSD=0)  
    简化版本，移除 FLAG 机制和分布式同步  
    """  
      
    def forward(self, model, sample, reduce=True):  
        """Compute the loss for the given sample."""  
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
              
            # 使用静态数据的归一化参数  
            targets_normalize = (targets - 7.044721989436028) / 1.352596540727069  
              
            # 计算 MSE loss  
            loss = nn.MSELoss(reduction="none")(  
                logits, targets_normalize[:logits.size(0)]  
            )  
            loss = (loss * weights).sum()  
              
            # 有效性检查  
            if not torch.isfinite(loss):  
                print(f"[CRITERION FP32] Non-finite loss detected: {loss.item()}", flush=True)  
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
            print(f"[CRITERION FP32] Exception: {str(e)}", flush=True)  
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