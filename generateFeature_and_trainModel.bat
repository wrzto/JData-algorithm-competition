@echo off
call python ConcatFile.py
call python TimeDecay.py
call python extract_feat_to_userModel.py
call python extract_feat_to_skuModel.py
call python xgb_userModel.py
call python xgb_skuModel.py
exit