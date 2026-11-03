1. 一行代表一篇文档。
2. 使用 jieba 精确模式，保留默认 HMM。
3. 先 Unicode NFKC 归一化，再把英文转小写。
4. 保留含字母或数字的 token，删除纯标点。
5. 第一版不删除停用词。
6. 空行保留为空文档。
7. 输出 UTF-8、无 BOM、LF 换行。
8. token 之间使用一个 ASCII 空格。
9. input.txt 输出为 input.tokens.txt。
