# 追加の環境構築ログ

## singularityの環境の中で下記を実行

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
git clone https://github.com/facebookresearch/detectron2.git
pip install detectron2
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install datasetutils
```
