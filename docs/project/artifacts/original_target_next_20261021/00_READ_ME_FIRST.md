# 接续入口

本轮120次原Target同轮比较与新方法筛选均已完成，不重复启动。先读[RESULTS.md](RESULTS.md)与[NEXT_METHODS.md](NEXT_METHODS.md)。主表分母统一为原Target，MLP4.637倍、EAGLEv3 2.986倍、v4复用2.888倍、Medusa2.454倍；包含执行优化的合计收益。

原/优化Target20/20输出和计数相同，四候选新旧主轮80/80相同，独立CPU复算10项通过。新树验证和DFlash尚未实现或训练，首选下一项为Feature W2真实分支成本与解码验证。当前原Target口径2倍已达，下一项应证明比当前MLP更快。

实际GPU入口：远端`/root/autodl-tmp/gaussiangpt_original_target_next_20261021/source_run01/evaluate_original_target.py`；结果`main_run01/`；本机源码`patches/original_target_next_20261021/`。生产及权重保留。
