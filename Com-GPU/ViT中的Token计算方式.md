$$
计算公式：N = (H/P  * W/P) + Extra
$$
H，W：输入图像的高度和宽度（如224）
P：PatchSize（如16）
Extra：额外添加的Special Token（如CLS）

> [!question] 那么ViT有多少个Token呢？

H，W = 224
P = 16
Extra = 1

N = (224/16 * 224/16) + 1 = 197个token