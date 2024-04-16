there are 2 post processing GLIP:

/cm/shared/kimth1/GLIP/maskrcnn_benchmark/modeling/rpn/inference.py 765 process nms and keep maximum the number of proposal objects as self.fpn_post_nms_top_n
/cm/shared/kimth1/GLIP/maskrcnn_benchmark/engine/predictor_glip.py 253 _post_process removes objects that have scores < threshold and sorts by descending scores

high score: 0.47
low score: 0.35