"""MacBERT 一步训练实验的外围：数据、加载、观测、日志与计时。"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import math
import statistics
import sys
import time
import textwrap

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT.parent / "models" / "chinese-macbert-base"
DEFAULT_TRAIN = Path("D:/Desktop/校/计算机/datasets/LCQMC/train.txt")
DEFAULT_MANIFEST = ROOT / "data" / "sample_manifest.json"


def has_pending_placeholder(source):
    """检测顺序骨架中可达的模板异常，忽略无条件 return 后的遗留文字。"""
    tree = ast.parse(textwrap.dedent(source))
    function = next(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    for statement in function.body:
        if isinstance(statement, ast.Return):
            return False
        if isinstance(statement, ast.Raise):
            error = statement.exc
            called = error.func if isinstance(error, ast.Call) else error
            if isinstance(called, ast.Name) and called.id == "NotImplementedError":
                return True
    return False


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    def encode_extra(item):
        if isinstance(item, set):
            return sorted(item)
        raise TypeError(f"不能保存为 JSON：{type(item).__name__}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=encode_extra) + "\n", encoding="utf-8")


def prepare_samples(source, target):
    """只从显式 train.txt 读取首批不同句对，固定取两类各两对。"""
    if source.name != "train.txt":
        raise ValueError("本实验只接受名为 train.txt 的训练源，不能使用 dev/test")
    selected = {0: [], 1: []}
    seen = set()
    with source.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 3 or fields[2] not in ("0", "1") or not all(fields[:2]):
                raise ValueError(f"train 第 {number} 行格式不符")
            a, b, label = fields[0], fields[1], int(fields[2])
            canonical = tuple(sorted((a, b)))
            if canonical in seen or len(selected[label]) == 2:
                continue
            seen.add(canonical)
            selected[label].append({"pair_id": f"train:{number - 1:06d}",
                                    "source_line": number, "text_a": a, "text_b": b, "label": label})
            if all(len(group) == 2 for group in selected.values()):
                break
    if not all(len(group) == 2 for group in selected.values()):
        raise ValueError("训练源中没有找到两类各两条不同句对")
    samples = [selected[label][i] for i in range(2) for label in (1, 0)]
    manifest = {"schema": 1, "split": "train", "source": str(source.resolve()),
                "source_sha256": file_hash(source), "selection": "first_two_unique_pairs_per_label_interleaved_1_0",
                "samples": samples}
    if target.exists():
        previous = json.loads(target.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("现有样本清单与训练源不一致；请核对原因，不自动覆盖")
    else:
        write_json(target, manifest)
    return manifest


def read_samples(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("split") != "train":
        raise ValueError("manifest 必须来自 train")
    source = Path(manifest["source"])
    if source.name != "train.txt" or file_hash(source) != manifest["source_sha256"]:
        raise ValueError("训练源名称或指纹不符")
    samples = manifest["samples"]
    if len(samples) != 4 or sorted(x["label"] for x in samples) != [0, 0, 1, 1]:
        raise ValueError("必须是四对、两类各两条样本")
    requested = {x["source_line"]: x for x in samples}
    if len(requested) != 4:
        raise ValueError("样本行号不能重复")
    verified = set()
    with source.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if number in requested:
                a, b, label = line.rstrip("\r\n").split("\t")
                item = requested[number]
                if (a, b, int(label), f"train:{number-1:06d}") != (
                        item["text_a"], item["text_b"], item["label"], item["pair_id"]):
                    raise ValueError(f"manifest 与 train 第 {number} 行不对应")
                verified.add(number)
            if len(verified) == 4:
                break
    if len(verified) != 4:
        raise ValueError("样本行号超出训练源")
    if len({tuple(sorted((x["text_a"], x["text_b"]))) for x in samples}) != 4:
        raise ValueError("样本清单中出现重复句对")
    return manifest


def validate_batch(batch, batch_size, vocab_size):
    import torch
    required = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    if set(batch) != required:
        raise ValueError(f"batch 字段不符：{set(batch)}")
    ids = batch["input_ids"]
    if batch_size <= 0 or ids.ndim != 2 or ids.shape[0] != batch_size or ids.shape[1] == 0:
        raise ValueError("input_ids 必须是非空 [B,L]")
    if any(t.dtype != torch.long or t.device.type != "cpu" for t in batch.values()):
        raise ValueError("本实验输入均为 CPU torch.long 张量")
    if any(batch[name].shape != ids.shape for name in ("attention_mask", "token_type_ids")):
        raise ValueError("mask、句段编号的形状必须与 ID 一致")
    if batch["labels"].shape != (batch_size,) or not set(batch["labels"].tolist()) <= {0, 1}:
        raise ValueError("labels 必须是 [B] 的 0/1 标签")
    if ids.min().item() < 0 or ids.max().item() >= vocab_size:
        raise ValueError("token ID 超出词表")
    for name in ("attention_mask", "token_type_ids"):
        if not set(batch[name].flatten().tolist()) <= {0, 1}:
            raise ValueError(f"{name} 必须只含 0/1")
    if (batch["attention_mask"].sum(dim=1) == 0).any():
        raise ValueError("不能出现全补齐样本")


def monitored_names(model):
    last = model.config.num_hidden_layers - 1
    return ["classifier.weight", "bert.encoder.layer.0.attention.self.query.weight",
            f"bert.encoder.layer.{last}.attention.self.query.weight"]


class StepObserver:
    """只观察调用边界，绝不替 student_step 执行前向、反向或更新。"""
    def __init__(self, model, optimizer, names, batch_size):
        import torch
        self.torch = torch
        self.model = model
        self.batch_size = batch_size
        self.parameters = dict(model.named_parameters())
        expected = {id(p) for p in self.parameters.values() if p.requires_grad}
        actual_list = [p for group in optimizer.param_groups for p in group["params"]]
        actual = {id(p) for p in actual_list}
        if expected != actual or len(actual_list) != len(actual):
            raise ValueError("优化器必须恰好持有全部待训练参数，且不能重复")
        if not all(p.requires_grad for p in self.parameters.values()):
            raise ValueError("首轮检验要求主干与分类头均不冻结")
        if any(group.get("weight_decay", 0) != 0 or group["lr"] <= 0 for group in optimizer.param_groups):
            raise ValueError("机制诊断要求 weight_decay=0、学习率为正")
        self.names = names
        self.before = {name: self.parameters[name].detach().clone() for name in names}
        self.stages = []
        self.output = None
        self.forward_count = 0
        self.backward_count = 0
        self.optimizer_step_count = 0
        self.captured_logits = None
        self.handles = [model.register_forward_hook(self._forward_hook),
                        optimizer.register_step_post_hook(self._step_hook)]
        self.started = self.last_time = time.perf_counter()

    def _forward_hook(self, module, inputs, output):
        self.forward_count += 1
        self.captured_logits = output if self.torch.is_tensor(output) else output.logits

    def _loss_hook(self, gradient):
        self.backward_count += 1

    def _step_hook(self, optimizer, args, kwargs):
        self.optimizer_step_count += 1

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __call__(self, stage, outputs):
        torch = self.torch
        expected = ["after_forward", "after_backward", "after_step"]
        if len(self.stages) >= 3 or stage != expected[len(self.stages)]:
            raise AssertionError(f"观测顺序应为 {expected}，本次收到 {stage}")
        if not self.model.training:
            raise AssertionError("训练步骤应设置 model.train()")
        if self.forward_count != 1 or outputs.logits is not self.captured_logits:
            raise AssertionError("必须恰好调用一次给定模型，并观察它返回的 logits")
        if stage == "after_forward" and (self.backward_count or self.optimizer_step_count):
            raise AssertionError("after_forward 之前不应已执行反向或更新")
        if stage in ("after_backward", "after_step") and self.backward_count != 1:
            raise AssertionError("应对本次 loss 恰好执行一次反向传播")
        expected_step_count = 1 if stage == "after_step" else 0
        if self.optimizer_step_count != expected_step_count:
            raise AssertionError("优化器更新次数或观测位置不符")
        if outputs.logits.shape != (self.batch_size, 2) or not torch.isfinite(outputs.logits).all():
            raise AssertionError("logits 必须是有限的 [B,2]")
        loss = outputs.loss
        if loss.ndim != 0 or not torch.isfinite(loss).item() or not loss.requires_grad or loss.grad_fn is None:
            raise AssertionError("loss 必须是保留计算关系的有限标量")
        if self.output is None:
            self.output = outputs
            self.handles.append(loss.register_hook(self._loss_hook))
        elif outputs is not self.output:
            raise AssertionError("各阶段必须观察同一次前向的 outputs")
        record = {"stage": stage, "loss": loss.item(), "seconds": time.perf_counter() - self.last_time,
                  "parameters": {}}
        for name in self.names:
            parameter = self.parameters[name]
            delta = (parameter.detach() - self.before[name]).abs().max().item()
            if not math.isfinite(delta):
                raise AssertionError(f"参数差值不是有限数：{name}")
            if stage != "after_step" and delta != 0:
                raise AssertionError(f"更新前参数不应变化：{name}")
            if stage == "after_forward":
                if parameter.grad is not None and torch.count_nonzero(parameter.grad).item() != 0:
                    raise AssertionError(f"前向后仍有旧梯度，请检查清理动作：{name}")
                norm = None
            else:
                grad = parameter.grad
                if grad is None or not torch.isfinite(grad).all():
                    raise AssertionError(f"梯度缺失或含非有限数：{name}")
                norm = grad.norm().item()
                if not math.isfinite(norm) or norm <= 0:
                    raise AssertionError(f"未观测到有效非零梯度：{name}")
                if stage == "after_step" and delta <= 0:
                    raise AssertionError(f"更新后没有观测到参数变化：{name}")
            record["parameters"][name] = {"max_abs_change": delta, "gradient_norm": norm}
        self.stages.append(record)
        self.last_time = time.perf_counter()

    def finish(self, outputs):
        if len(self.stages) != 3 or outputs is not self.output:
            raise AssertionError("没有完整记录三阶段，或没有返回同一次前向的 outputs")
        if (self.forward_count, self.backward_count, self.optimizer_step_count) != (1, 1, 1):
            raise AssertionError("每一步都应只前向、反向、更新各一次")
        self.close()
        return {"seconds": time.perf_counter() - self.started, "stages": self.stages,
                "forward_count": self.forward_count, "backward_count": self.backward_count,
                "optimizer_step_count": self.optimizer_step_count}


def prepare_runtime(args):
    import torch
    import transformers
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    config = BertConfig.from_pretrained(str(args.model_dir), local_files_only=True)
    if (config.num_hidden_layers, config.hidden_size) != (12, 768):
        raise ValueError("请选择 MacBERT-base 的 12 层、768 维配置，当前目录不是本实验规格")
    provenance_file = args.model_dir / "download_manifest.json"
    if not provenance_file.exists():
        raise ValueError("缺少固定版本下载清单，请先运行 prepare_model.py")
    provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
    from prepare_model import REPO, REVISION, FILES
    if (provenance.get("repo"), provenance.get("revision")) != (REPO, REVISION):
        raise ValueError("模型来源或固定版本与本实验不一致")
    if set(provenance.get("files", {})) != set(FILES):
        raise ValueError("下载清单没有恰好覆盖所需文件")
    for name, fingerprint in provenance["files"].items():
        if file_hash(args.model_dir / name) != fingerprint["sha256"]:
            raise ValueError(f"模型文件指纹改变：{name}")
    start = time.perf_counter()
    tokenizer = BertTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    config.num_labels = 2
    model, loading = BertForSequenceClassification.from_pretrained(
        str(args.model_dir), config=config, local_files_only=True,
        output_loading_info=True, attn_implementation="eager")
    load_seconds = time.perf_counter() - start
    # MLM 检查点没有任务分类头，pooler 也可能尚未初始化；编码器主干不允许缺失。
    allowed_missing = ("classifier.", "bert.pooler.")
    missing_backbone = [key for key in loading.get("missing_keys", []) if not key.startswith(allowed_missing)]
    if missing_backbone or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise ValueError(f"预训练主干加载不完整：{loading}")
    manifest = read_samples(args.manifest)
    samples = manifest["samples"][:args.batch_size]
    encoded = tokenizer([x["text_a"] for x in samples], [x["text_b"] for x in samples],
                        padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
    batch = dict(encoded)
    batch["labels"] = torch.tensor([x["label"] for x in samples], dtype=torch.long)
    validate_batch(batch, len(samples), config.vocab_size)
    full_lengths = [len(tokenizer(x["text_a"], x["text_b"], truncation=False)["input_ids"]) for x in samples]
    report = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
              "device": "cpu", "seed": 42, "threads": args.threads, "repo": provenance["repo"],
              "revision": provenance["revision"], "manifest_sha256": file_hash(args.manifest),
              "model_load_seconds": load_seconds, "parameter_count": sum(p.numel() for p in model.parameters()),
              "model_dir": str(args.model_dir.resolve()), "max_length": args.max_length,
              "batch_shapes": {k: list(v.shape) for k, v in batch.items()},
              "pair_ids": [x["pair_id"] for x in samples], "labels": batch["labels"].tolist(),
              "batch_preview": {key: value.tolist() for key, value in batch.items()},
              "tokens": [tokenizer.convert_ids_to_tokens(row) for row in batch["input_ids"].tolist()],
              "untruncated_lengths": full_lengths, "truncated": [n > args.max_length for n in full_lengths],
              "loading_info": loading}
    return torch, model, batch, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "preflight", "train"), required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1, help="首步验收后可设6：一轮热身、五次稳定步")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 4 or not 8 <= args.max_length <= 128 or not 1 <= args.steps <= 6 or args.threads < 1:
        parser.error("batch-size 为1..4，max-length 为8..128，steps 为1..6，threads 为正数")
    if args.stage == "prepare":
        manifest = prepare_samples(args.train_file, args.manifest)
        print(f"SAMPLES_READY count=4 labels=2+2 manifest={args.manifest}")
        print("pair_ids=" + ",".join(x["pair_id"] for x in manifest["samples"]))
        return
    if args.stage == "train":
        import inspect
        from student_step import train_one_step
        if has_pending_placeholder(inspect.getsource(train_one_step)):
            print("STUDENT_TODO: 先完成 student_step.py 的 train_one_step；未开始加载或训练。")
            raise SystemExit(2)
    torch, model, batch, report = prepare_runtime(args)
    if args.stage == "preflight":
        names = monitored_names(model)
        before = {n: dict(model.named_parameters())[n].detach().clone() for n in names}
        model.eval()
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**batch)
        if outputs.logits.shape != (args.batch_size, 2) or not torch.isfinite(outputs.logits).all() or not torch.isfinite(outputs.loss):
            raise AssertionError("预检输出形状或数值不符")
        if any(not torch.equal(before[n], dict(model.named_parameters())[n]) for n in names):
            raise AssertionError("预检前向不应改变被观测参数")
        report.update({"status": "PREFLIGHT_OK", "training_executed": False,
                       "forward_seconds": time.perf_counter() - start,
                       "logits_shape": list(outputs.logits.shape), "loss": outputs.loss.item(),
                       "monitored_parameters_unchanged": True})
        write_json(args.output_dir / "preflight.json", report)
        print("PREFLIGHT_OK logits=" + str(list(outputs.logits.shape)) + " no_parameter_update=True")
        print("尚未执行训练；请填写 student_step.py。loss 不代表模型效果。")
        return
    from student_step import train_one_step
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.0)
    report.update({"learning_rate": 2e-5, "weight_decay": 0.0,
                   "training_attempted": True, "training_executed": False,
                   "student_step_sha256": file_hash(ROOT / "student_step.py"), "steps": []})
    try:
        for index in range(args.steps):
            observer = StepObserver(model, optimizer, monitored_names(model), args.batch_size)
            outputs = train_one_step(model, batch, optimizer, observer)
            result = observer.finish(outputs)
            report["steps"].append(result)
            report["training_executed"] = True
            print(f"UPDATE_OK step={index + 1} seconds={result['seconds']:.3f}", flush=True)
        if args.steps > 1:
            report["median_seconds_excluding_first"] = statistics.median(x["seconds"] for x in report["steps"][1:])
        report["status"] = "UPDATE_VERIFIED"
        write_json(args.output_dir / "training_check.json", report)
        print("UPDATE_VERIFIED: 梯度与参数变化检查通过；不是论文准确率复现。")
    except Exception as exc:
        report.update({"status": "FAILED", "error": str(exc), "error_type": type(exc).__name__})
        if "observer" in locals():
            report["last_observations"] = observer.stages
            report["observed_optimizer_steps_in_last_attempt"] = observer.optimizer_step_count
            report["training_executed"] = bool(report["steps"] or observer.optimizer_step_count)
            observer.close()
        write_json(args.output_dir / "training_failure.json", report)
        raise


if __name__ == "__main__":
    main()
