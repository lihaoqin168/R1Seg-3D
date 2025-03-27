# from pycrawlers import huggingface
#
# # 实例化类
# hg = huggingface()
#
# urls = ['https://huggingface.co/datasets/SpamYdob/AbdomenAtlas3.0Report/tree/main',
#         'https://huggingface.co/datasets/SpamYdob/AbdomenAtlasData/tree/main']
#
# # 批量下载
# # 默认保存位置在当前脚本所在文件夹 ./
# # hg.get_batch_data(urls)
#
# # 自定义下载位置
# paths = ['E:/02datasets/AbdomenAtlas3.0Report/', 'E:/02datasets/AbdomenAtlasData/']
# hg.get_batch_data(urls, paths)

#
# cd E:/02datasets/AbdomenAtlas3.0Report/
#
# huggingface-cli download SpamYdob/AbdomenAtlas3.0Report --token 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 --repo-type dataset --local-dir .
#
# huggingface-cli download SpamYdob/AbdomenAtlasData --token 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 --repo-type dataset --local-dir .

##########################

#
# https://huggingface.co/docs/hub/security-git-ssh
#
# ssh -T git@hf.co

# #datasets
# git clone git@hf.co:datasets/SpamYdob/AbdomenAtlasData
# git clone git@hf.co:datasets/SpamYdob/AbdomenAtlas3.0Report

# #model
# git clone git@hf.co:SpamYdob/AbdomenAtlasData
