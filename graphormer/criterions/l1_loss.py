from fairseq.dataclass.configs import FairseqDataclass  
  
import torch  
import torch.nn as nn  
from fairseq import metrics  
from fairseq.criterions import FairseqCriterion, register_criterion  
import numpy as np  
  
torch.autograd.set_detect_anomaly(True)  
  
def check_tensor_for_nan(tensor, name, sample_id=None):  
    """Helper function to check for NaN/Inf values in tensors"""  
    if isinstance(tensor, torch.Tensor):  
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():  
            sample_info = f" in sample {sample_id}" if sample_id is not None else ""  
            # print(f"WARNING: NaN/Inf detected in {name}{sample_info}")  
            return True  
    return False  
  
@register_criterion("l1_loss", dataclass=FairseqDataclass)  
class GraphPredictionL1Loss(FairseqCriterion):  
    """  
    Implementation for the L1 loss (MAE loss) used in graphormer model training.  
    """  
  
    def forward(self, model, sample, reduce=True):  
        """Compute the loss for the given sample.  
  
        Returns a tuple with three elements:  
        1) the loss  
        2) the sample size, which is used as the denominator for the gradient  
        3) logging outputs to display while training  
        """  
        sample_size = sample["nsamples"]  
          
        # Print sample ID if available  
        sample_id = None  
        if "pdbid" in sample:  
            sample_id = sample["pdbid"]  
            print(f"Processing sample with ID: {sample_id}")  
          
        # Check input data for NaN values  
        for key, value in sample["net_input"].items():  
            if isinstance(value, dict):  
                for subkey, subvalue in value.items():  
                    check_tensor_for_nan(subvalue, f"input[{key}][{subkey}]", sample_id)  
            else:  
                check_tensor_for_nan(value, f"input[{key}]", sample_id)  
  
        with torch.no_grad():  
            natoms = sample["net_input"]["batched_data"]["x"].shape[1]  
            # print(f"Number of atoms in batch: {natoms}")  
  
        try:  
            # Forward pass  
            logits = model(**sample["net_input"])  
              
            # Check model output for NaN  
            if isinstance(logits, tuple):  
                logits, weights = logits  
                check_tensor_for_nan(weights, "model weights", sample_id)  
            else:  
                weights = torch.ones(logits.shape, dtype=logits.dtype, device=logits.device)  
              
            check_tensor_for_nan(logits, "model output (logits)", sample_id)  
              
            targets = model.get_targets(sample, [logits])  
            check_tensor_for_nan(targets, "targets", sample_id)  
              
            # md data  
            targets_normalize = (targets - 6.529300030461668) / 1.9919705951218716  
            check_tensor_for_nan(targets_normalize, "normalized targets", sample_id)  
  
            # Loss calculation  
            loss_per_sample = nn.L1Loss(reduction="none")(logits, targets_normalize[: logits.size(0)])  
            check_tensor_for_nan(loss_per_sample, "loss per sample", sample_id)  
              
            loss = (loss_per_sample * weights).sum()  
            if check_tensor_for_nan(loss, "final loss", sample_id):  
                print(f"WARNING: NaN in final loss! Sample details:")  
                print(f"  - Sample ID: {sample_id}")  
                print(f"  - Logits min/max/mean: {logits.min().item():.4f}/{logits.max().item():.4f}/{logits.mean().item():.4f}")  
                print(f"  - Targets min/max/mean: {targets.min().item():.4f}/{targets.max().item():.4f}/{targets.mean().item():.4f}")  
                print(f"  - Loss per sample min/max/mean: {loss_per_sample.min().item():.4f}/{loss_per_sample.max().item():.4f}/{loss_per_sample.mean().item():.4f}")  
  
            logging_output = {  
                "loss": loss.data,  
                "sample_size": logits.size(0),  
                "nsentences": sample_size,  
                "ntokens": natoms,  
            }  
            return loss, sample_size, logging_output  
              
        except Exception as e:  
            print(f"ERROR during forward pass: {str(e)}")  
            print(f"Sample ID: {sample_id}")  
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
  
  
@register_criterion("l1_loss_with_flag", dataclass=FairseqDataclass)  
class GraphPredictionL1LossWithFlag(GraphPredictionL1Loss):  
    """  
    Implementation for the binary log loss used in graphormer model training.  
    """  
  
    def forward(self, model, sample, reduce=True):  
        """Compute the loss for the given sample.  
  
        Returns a tuple with three elements:  
        1) the loss  
        2) the sample size, which is used as the denominator for the gradient  
        3) logging outputs to display while training  
        """  
        sample_size = sample["nsamples"]  
        perturb = sample.get("perturb", None)  
          
        # Print sample ID if available  
        sample_id = None  
        if "pdbid" in sample:  
            sample_id = sample["pdbid"]  
            print(f"Processing sample with ID: {sample_id}")  
          
        # Check input data for NaN values  
        for key, value in sample["net_input"].items():  
            if isinstance(value, dict):  
                for subkey, subvalue in value.items():  
                    check_tensor_for_nan(subvalue, f"input[{key}][{subkey}]", sample_id)  
            else:  
                check_tensor_for_nan(value, f"input[{key}]", sample_id)  
  
        batch_data = sample["net_input"]["batched_data"]["x"]  
        with torch.no_grad():  
            natoms = batch_data.shape[1]  
            # print(f"Number of atoms in batch: {natoms}")  
  
        try:  
            # Forward pass with perturb parameter  
            logits = model(**sample["net_input"], perturb=perturb)  
              
            # Check model output for NaN  
            if isinstance(logits, tuple):  
                logits, weights = logits  
                check_tensor_for_nan(weights, "model weights", sample_id)  
            else:  
                weights = torch.ones(logits.shape, dtype=logits.dtype, device=logits.device)  
              
            check_tensor_for_nan(logits, "model output (logits)", sample_id)  
              
            targets = model.get_targets(sample, [logits])  
            check_tensor_for_nan(targets, "targets", sample_id)  
              
            # md data  
            targets_normalize = (targets - 6.529300030461668) / 1.9919705951218716  
            check_tensor_for_nan(targets_normalize, "normalized targets", sample_id)  
  
            # Loss calculation  
            loss_per_sample = nn.L1Loss(reduction="none")(logits, targets_normalize[: logits.size(0)])  
            check_tensor_for_nan(loss_per_sample, "loss per sample", sample_id)  
              
            loss = (loss_per_sample * weights).sum()  
            if check_tensor_for_nan(loss, "final loss", sample_id):  
                print(f"WARNING: NaN in final loss! Sample details:")  
                print(f"  - Sample ID: {sample_id}")  
                print(f"  - Logits min/max/mean: {logits.min().item():.4f}/{logits.max().item():.4f}/{logits.mean().item():.4f}")  
                print(f"  - Targets min/max/mean: {targets.min().item():.4f}/{targets.max().item():.4f}/{targets.mean().item():.4f}")  
                print(f"  - Loss per sample min/max/mean: {loss_per_sample.min().item():.4f}/{loss_per_sample.max().item():.4f}/{loss_per_sample.mean().item():.4f}")  
                if perturb is not None:  
                    print(f"  - Using perturbation: {perturb}")  
  
            logging_output = {  
                "loss": loss.data,  
                "sample_size": logits.size(0),  
                "nsentences": sample_size,  
                "ntokens": natoms,  
            }  
            return loss, sample_size, logging_output  
              
        except Exception as e:  
            print(f"ERROR during forward pass with FLAG: {str(e)}")  
            print(f"Sample ID: {sample_id}")  
            if perturb is not None:  
                print(f"Using perturbation: {perturb}")  
            raise e