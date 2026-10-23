# Todo
- [x] Research competition (metric = PQ, COCO polygons, 707 images / 1154 annotator records)
- [x] Build YOLO11-seg train → PQ hold-out → submission notebook
- [x] Local smoke test (labels, metric 1.0/0.0 sanity, inference + RLE)
- [x] Private GitHub repo
- [ ] Full GPU run on Kaggle, record hold-out PQ, submit
- [ ] Ideas: tiled 2048 inference, flip TTA w/ instance matching, 5-fold ensemble (WBF on masks), Mask R-CNN / crop U-Net refiner
