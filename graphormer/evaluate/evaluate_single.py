import torch  
from torchmetrics import functional as MF  
from fairseq import utils, options, tasks  
from fairseq.logging import progress_bar  
from fairseq.dataclass.utils import convert_namespace_to_omegaconf  
import numpy as np  
import pandas as pd  
import logging  
from pathlib import Path  
import sys  
  
logger = logging.getLogger(__name__)  
  
def mean_frame(df_one):  
    """计算每个 PDB ID 的多个 frame 的平均预测值"""  
    return df_one['y_pred'].mean()  
  
def gen_result(df, test_id, func):  
    """对每个 PDB ID 应用聚合函数并生成最终结果"""  
    pdbid, true, pred = [], [], []  
    for id in test_id:  
        res = df[df['pdbid'] == id]  
        p = func(res)  
        if p < 0:     
            continue  
        pdbid.append(id)  
        true.append(res['y_true'].to_list()[0])  
        pred.append(p)  
        
    pdbid, true, pred = np.array(pdbid), np.array(true), np.array(pred)  
    idx = true.argsort()  
    pdbid, true, pred = pdbid[idx], true[idx], pred[idx]  
    return pdbid, true, pred  
  
def eval(args, cfg, task, model, checkpoint_path):  
    """评估单个 checkpoint"""  
    save_file = checkpoint_path.parent / (checkpoint_path.name.split('.')[0] + f'{args.suffix}' + '.csv')  
        
    logger.info(f"Loading checkpoint from {checkpoint_path}")  
    model_state = torch.load(checkpoint_path, weights_only=False)["model"]  
    model.load_state_dict(model_state, strict=True, model_cfg=cfg.model)  
    del model_state  
    model.to(torch.cuda.current_device())  
        
    split = args.split  
    task.load_dataset(split)  
    batch_iterator = task.get_batch_iterator(  
        dataset=task.dataset(split),  
        max_tokens=cfg.dataset.max_tokens_valid,  
        max_sentences=cfg.dataset.batch_size_valid,  
        max_positions=utils.resolve_max_positions(  
            task.max_positions(),  
            model.max_positions(),  
        ),  
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,  
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,  
        seed=cfg.common.seed,  
        num_workers=cfg.dataset.num_workers,  
        epoch=0,  
        data_buffer_size=cfg.dataset.data_buffer_size,  
        disable_iterator_cache=False,  
    )  
        
    itr = batch_iterator.next_epoch_itr(shuffle=False, set_dataset_epoch=False)  
    progress_bar_obj = progress_bar.progress_bar(  
        itr,  
        log_format=cfg.common.log_format,  
        log_interval=cfg.common.log_interval,  
    )  
  
    y_pred = []  
    y_true = []  
    pdbid_list = []  
        
    with torch.no_grad():  
        model.eval()  
        for i, sample in enumerate(progress_bar_obj):  
            sample = utils.move_to_cuda(sample)  
                
            # 模型预测  
            y = model(**sample["net_input"])  
            if isinstance(y, tuple):     
                y = y[0]  
            y = y.reshape(-1)  
            y_pred.extend(y.detach().cpu())  
                
            # 获取真实标签  
            targets = sample["target"]  
            y_true.extend(targets.cpu().reshape(-1))  
                
            # 获取 pdbid  
            batched_data = sample["net_input"]["batched_data"]  
            pdbid_list.extend(batched_data['pdbid'])  
                
            torch.cuda.empty_cache()  
  
    # 转换为 tensor/array  
    y_pred = torch.Tensor(y_pred)  
    y_true = torch.Tensor(y_true)  
        
    # 反标准化 - 修改为乘以5.0以匹配训练时的标准化方式  
    y_pred = y_pred * 5.0  
        
    # 保存详细结果 (不包含 frame 字段)  
    logger.info(f"Saving results to {save_file}")  
    df = pd.DataFrame({  
        "pdbid": pdbid_list,  
        "y_true": y_true.to(torch.float32).cpu().numpy(),  
        "y_pred": y_pred.to(torch.float32).cpu().numpy(),  
    })  
    df.to_csv(save_file, index=False)  
        
    # 直接计算指标,不需要对 frame 取平均  
    pearson_r = MF.pearson_corrcoef(y_pred.to(torch.float32), y_true.to(torch.float32))  
    logger.info(f"Pearson correlation coefficient (Rp): {pearson_r:.4f}")  
        
    rmse = torch.sqrt(MF.mean_squared_error(y_pred.to(torch.float32), y_true.to(torch.float32)))  
    logger.info(f"RMSE: {rmse:.4f}")  
        
    mae = np.mean(np.abs(y_true.numpy() - y_pred.numpy()))  
    logger.info(f"MAE: {mae:.4f}")  
        
    logger.info(f"Total samples: {len(df)}")  
    logger.info(f"Unique PDB IDs: {df['pdbid'].nunique()}")  
  
def main():  
    parser = options.get_training_parser()  
    parser.add_argument("--split", type=str, required=True, help="Dataset split to evaluate (e.g., test)")  
    parser.add_argument("--suffix", type=str, default="", help="Suffix for output CSV file")  
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to specific checkpoint file")  
    
    args = options.parse_args_and_arch(parser, modify_parser=None)  
    cfg = convert_namespace_to_omegaconf(args)  
        
    # 设置随机种子  
    np.random.seed(cfg.common.seed)  
    utils.set_torch_seed(cfg.common.seed)  
        
    # 初始化任务和模型  
    task = tasks.setup_task(cfg.task)  
    model = task.build_model(cfg.model)  
        
    # 验证 checkpoint 路径  
    checkpoint_path = Path(args.checkpoint_path)  
    if not checkpoint_path.exists():  
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")  
        
    logger.info(f"Evaluating checkpoint: {checkpoint_path}")  
    eval(args, cfg, task, model, checkpoint_path)  
    sys.stdout.flush()  
  
if __name__ == '__main__':  
    main()