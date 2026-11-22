"""验证外围工具能拒绝坏数据和伪更新。这里不实现用户的训练函数。"""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from experiment import (StepObserver, prepare_samples, read_samples, validate_batch, write_json,
                        has_pending_placeholder)


class PlaceholderChecks(unittest.TestCase):
    def test_unfilled_placeholder_is_pending(self):
        self.assertTrue(has_pending_placeholder('def step():\n    raise NotImplementedError("todo")\n'))

    def test_placeholder_after_return_is_unreachable(self):
        self.assertFalse(has_pending_placeholder('def step():\n    return result\n    raise NotImplementedError("todo")\n'))

    def test_mention_in_docstring_does_not_block(self):
        source = 'def step():\n    """remove raise NotImplementedError( when done"""\n    return result\n'
        self.assertFalse(has_pending_placeholder(source))


class DataChecks(unittest.TestCase):
    def batch(self):
        return {"input_ids": torch.tensor([[1, 2], [3, 0]]),
                "attention_mask": torch.tensor([[1, 1], [1, 0]]),
                "token_type_ids": torch.tensor([[0, 1], [0, 0]]),
                "labels": torch.tensor([1, 0])}

    def test_normal_batch(self):
        validate_batch(self.batch(), 2, 10)

    def test_label_shape_rejected(self):
        batch = self.batch()
        batch["labels"] = batch["labels"].reshape(2, 1)
        with self.assertRaises(ValueError):
            validate_batch(batch, 2, 10)

    def test_invalid_label_rejected(self):
        batch = self.batch()
        batch["labels"][0] = 2
        with self.assertRaises(ValueError):
            validate_batch(batch, 2, 10)

    def test_float_ids_rejected(self):
        batch = self.batch()
        batch["input_ids"] = batch["input_ids"].float()
        with self.assertRaises(ValueError):
            validate_batch(batch, 2, 10)

    def test_empty_batch_rejected(self):
        batch = {key: value[:0] for key, value in self.batch().items()}
        with self.assertRaises(ValueError):
            validate_batch(batch, 0, 10)

    def test_manifest_roundtrip_and_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "train.txt"
            source.write_text("a\tb\t1\nc\td\t0\ne\tf\t1\ng\th\t0\n", encoding="utf-8")
            target = Path(folder) / "manifest.json"
            expected = prepare_samples(source, target)
            self.assertEqual(read_samples(target), expected)
            source.write_text("changed\tb\t1\nc\td\t0\ne\tf\t1\ng\th\t0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_samples(target)

    def test_dev_source_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "dev.txt"
            source.write_text("a\tb\t1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_samples(source, Path(folder) / "manifest.json")

    def test_insufficient_classes_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "train.txt"
            source.write_text("a\tb\t1\nc\td\t1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_samples(source, Path(folder) / "manifest.json")

    def test_loading_report_sets_serializable(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "loading.json"
            write_json(target, {"missing_keys": {"classifier.weight", "classifier.bias"}})
            self.assertIn("classifier.weight", target.read_text(encoding="utf-8"))


class ObserverChecks(unittest.TestCase):
    def fixture(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0)
        names = ["0.weight", "1.weight"]
        observer = StepObserver(model, optimizer, names, 2)
        logits = model(torch.ones(2, 2))
        outputs = SimpleNamespace(logits=logits, loss=logits.square().mean())
        return model, optimizer, observer, outputs

    @staticmethod
    def synthetic_gradients(model):
        # 仅注入观测器测试信号，不在这里反向训练，不调用 student_step。
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)

    def test_incomplete_optimizer_rejected(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        optimizer = torch.optim.AdamW(model[1].parameters(), weight_decay=0)
        with self.assertRaises(ValueError):
            StepObserver(model, optimizer, ["0.weight", "1.weight"], 2)

    def test_wrong_stage_order_rejected(self):
        _, _, observer, outputs = self.fixture()
        with self.assertRaises(AssertionError):
            observer("after_step", outputs)

    def test_detached_loss_rejected(self):
        _, _, observer, outputs = self.fixture()
        outputs.loss = outputs.loss.detach()
        with self.assertRaises(AssertionError):
            observer("after_forward", outputs)

    def test_missing_gradient_rejected(self):
        _, _, observer, outputs = self.fixture()
        observer("after_forward", outputs)
        with self.assertRaises(AssertionError):
            observer("after_backward", outputs)

    def test_nan_gradient_rejected(self):
        model, _, observer, outputs = self.fixture()
        observer("after_forward", outputs)
        outputs.loss.backward()
        model[0].weight.grad[0, 0] = float("nan")
        with self.assertRaises(AssertionError):
            observer("after_backward", outputs)

    def test_no_update_rejected(self):
        model, _, observer, outputs = self.fixture()
        observer("after_forward", outputs)
        outputs.loss.backward()
        observer("after_backward", outputs)
        with self.assertRaises(AssertionError):
            observer("after_step", outputs)

    def test_early_change_rejected_and_snapshot_independent(self):
        model, _, observer, outputs = self.fixture()
        with torch.no_grad():
            model[0].weight.add_(0.1)
        with self.assertRaises(AssertionError):
            observer("after_forward", outputs)

    def test_tiny_module_observation_path(self):
        # 只验证观测器支持真实 autograd/optimizer hooks，不调用学生函数或 MacBERT。
        model, optimizer, observer, outputs = self.fixture()
        observer("after_forward", outputs)
        outputs.loss.backward()
        observer("after_backward", outputs)
        optimizer.step()
        observer("after_step", outputs)
        result = observer.finish(outputs)
        self.assertEqual(len(result["stages"]), 3)
        self.assertGreater(result["stages"][-1]["parameters"]["0.weight"]["max_abs_change"], 0)

    def test_manually_fabricated_gradients_rejected(self):
        model, _, observer, outputs = self.fixture()
        observer("after_forward", outputs)
        self.synthetic_gradients(model)
        with self.assertRaises(AssertionError):
            observer("after_backward", outputs)

    def test_extra_forward_rejected(self):
        model, _, observer, outputs = self.fixture()
        model(torch.ones(2, 2))
        with self.assertRaises(AssertionError):
            observer("after_forward", outputs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
