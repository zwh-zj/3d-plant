# 3d-plant

Crops3D 文本查询器官分割。仓库只保留源码。`pointnet2/_ext*.so` 和 `gres_model/lib/pointgroup_ops/pointgroup_ops_ext*.so` 是本机 CUDA 编译结果，不入库。克隆后需要在本机重新编译。

## 编译扩展

用即将运行训练的那个 Python。需要已安装带 CUDA 的 PyTorch，并且 `nvcc` 和 `g++` 在 `PATH` 中。先确认：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

输出必须是 `True`。`gres_model/lib/setup.py` 在 CUDA 不可用时不会生成扩展。

在仓库根目录执行：

```bash
cd pointnet2
python setup.py build_ext --inplace
cd ../gres_model/lib
python setup.py build_ext --inplace
```

`--inplace` 把扩展放在源码旁边：

- `pointnet2/_ext*.so`
- `gres_model/lib/pointgroup_ops/pointgroup_ops_ext*.so`

同时会生成 `pointnet2/build/` 和 `gres_model/lib/build/`。这些目录和 `.so` 都不要提交。文件名里的 `cpython-38` 随编译时使用的 Python 版本变化，换 Python 或换 CUDA 后要重编。

`import pointgroup_ops` 需要把 `gres_model/lib` 放进 `PYTHONPATH`。在仓库根目录：

```bash
export PYTHONPATH="$(pwd):$(pwd)/gres_model/lib${PYTHONPATH:+:$PYTHONPATH}"
```

在 WSL 上跑 GPU 时，再加上：

```bash
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```
