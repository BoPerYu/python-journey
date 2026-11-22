"""本人只在这个文件串起训练数据流。外围验证不在这里实现。"""


def train_one_step(model, batch, optimizer, observe):
    """执行且仅执行一次训练更新，返回模型输出 outputs。

    输入
    ----
    model: 已加载预训练主干并接上随机初始化二分类头的 PyTorch 模型。
    batch: 字典，包含 input_ids、attention_mask、token_type_ids、labels。
           前三个通常形状为 [B, L]，labels 为 [B]，dtype 为 torch.long。
    optimizer: 已持有全部待训练参数的优化器。
    observe: 外围观察函数，按下面的阶段调用，不负责计算或更新参数。

    输出
    ----
    outputs: 本次前向的模型输出对象，带有 outputs.logits 和 outputs.loss。

    不变量与边界
    ----------
    - batch 已由外围保证非空、标签为 0/1；不要修改样本或标签。
    - 本函数只进行一次前向、一次反向、一次优化器更新。
    - 本轮主干和分类头都参与训练；不要冻结或替换模型。
    - loss 必须保留梯度关系，不能先转成 Python 数字或 detach 再反向。
    - 不需要手动 softmax，传入 labels 后模型会计算分类损失。

    查阅提示（这是 API 说明，不是待抄写的完整实现）
    -------------------------------------------
    model.train(): 设置训练模式，不等于执行训练。
    optimizer.zero_grad(set_to_none=True): 清理旧梯度，不更新权重。
    model(**batch): 将字典按参数名传给模型，返回 outputs。
    outputs.loss: 本次分类损失；outputs.logits: 每条句对的两个类别分数。
    Tensor.backward(): 计算梯度。
    optimizer.step(): 执行参数更新。

    TODO 按以下顺序填写，完成后删掉下面的 raise
    --------------------------------------
    1. 设置训练模式、清理旧梯度。
    2. 把 batch 交给 model，将返回对象命名为 outputs。
    3. 调用 observe("after_forward", outputs)。
    4. 用 outputs.loss 触发反向传播。
    5. 调用 observe("after_backward", outputs)。
    6. 让 optimizer 更新一次参数。
    7. 调用 observe("after_step", outputs)。
    8. 返回 outputs。
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(**batch)
    observe("after_forward",outputs)
    outputs.loss.backward()
    observe("after_backward",outputs)
    optimizer.step()
    observe("after_step",outputs)
    return outputs

    raise NotImplementedError("请先在 student_step.py 中完成一次训练的数据流")
