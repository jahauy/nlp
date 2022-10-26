# -*- coding: utf-8 -*-
# @Author: Edison
# @Date: 2022/10/24

import os, codecs
from jieba import lcut
from snownlp import *
import numpy as np
from wordcloud import WordCloud
from PIL import Image
from matplotlib import pyplot as plt
from docx import Document


def _read_txt(path):
    res = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            res.append(line.rstrip())
    return res

def _read_docx(path):
    pgs = Document(path).paragraphs
    res = ''
    for p in pgs:
        res += (p.text + ' ')
    return res

def read_target(root_path, filename):
    path = os.path.join(root_path, filename)
    if filename.endswith('docx'):
        return _read_docx(path)
    else:
        return ' '.join(_read_txt(path))

def get_stop_words(root_path, filename='stopwords.txt'):
    path = os.path.join(root_path, filename)
    if not os.path.exists(path):
        return None
    return _read_txt(path)
    
def _seg(docs):
    return lcut(docs)

def _preprocess_words(sent):
    words = _seg(sent)
    words = normal.filter_stop(words)
    return words

def _get_datasets(path, filename):
    file = os.path.join(path, filename)
    return codecs.open(file, 'r', 'utf-8').readlines()


class SentiCls(sentiment.Sentiment):
    def __init__(self):
        super().__init__()
        
    def handle(self, doc):
        return _preprocess_words(doc)

    
class Analysis(SnowNLP):
    def __init__(self, docs):
        super().__init__(docs)
        self.cls = None
        
    @property
    def words(self):
        return seg(self.doc)
    
    @property
    def sentiment(self):
        return self.cls.classify(self.doc)
    
    def set_classifier(self, cls):
        self.cls = cls
        
    def _preprocess(self):
        doc = []
        for sent in self.sentences:
            words = _preprocess_words(sent)
            doc.append(words)
        return doc
    
    def keywords(self, limit=5, merge=False):
        doc = self._preprocess()   
        rank = textrank.KeywordTextRank(doc)
        rank.solve()
        ret = []
        for w in rank.top_index(limit):
            ret.append(w)
        if merge:
            wm = words_merge.SimpleMerge(self.doc, ret)
            return wm.merge()
        return ret
    
    def summary(self, limit=5):
        doc = self._preprocess()   
        rank = textrank.TextRank(doc)
        rank.solve()
        ret = []
        for index in rank.top_index(limit):
            ret.append(self.sentences[index])
        return ret
    
    def gen_wordcloud(self, dst_path, wc_bg=None, stopwords=None, show=True):
        mask = np.array(Image.open(wc_bg)) if not wc_bg is None else None
        wc = WordCloud(
            font_path='msyh.ttc',
            background_color='white',
            max_words=1000,
            max_font_size=100,
            stopwords=stopwords,
            mask=mask
        )
        wc.generate(' '.join(self.keywords(200)))
        wc.to_file(dst_path)
        
        if show:
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.show()
            
            
class Inference(object):
    def __init__(self, docs, dst_path=os.getcwd()):
        self.anl = Analysis(docs)
        self.model_path = os.path.join(dst_path, 'sentiment.marshal.3')
        self.wc_path = os.path.join(dst_path, 'result.jpg')
    
    def _train(self, classifier, path, save):
        print('Start training...')
        neg, pos = _get_datasets(path, 'neg.txt'), _get_datasets(path, 'pos.txt')
        classifier.train(neg, pos)
        if save:
            classifier.save(self.model_path)
        print('Training finished.')
    
    def _load(self, dataset_path, train, save):
        classifier = SentiCls()
        if train:
            self._train(classifier, dataset_path, save)
        classifier.load(self.model_path)
        return classifier
    
    def __call__(self, dataset_path, train=True, save=True, 
                 wc_bg=None, stop_words=None, wc_show=True):
        self.anl.set_classifier(self._load(dataset_path, train=train, save=save))
        self.anl.gen_wordcloud(self.wc_path, 
                               wc_bg=wc_bg, stopwords=stop_words, show=wc_show)
        def _log(title, infos):
            print(title, end='：')
            for i in infos:
                print(f'{i}', end='. ')
            print('')
        print(f'情绪指数：{self.anl.sentiments}')
        _log('关键词', self.anl.keywords())
        _log('摘要', self.anl.summary())
    
    
def main(wd, dataset_path):
    text = read_target(wd, '二十大报告.docx')
    model = Inference(text, dst_path=wd)
    model(dataset_path)



if __name__ == '__main__':
    work_dir = r'D:\workspace\projects\nlp\static'
    dataset_path = r'D:\softwares\program_tools\anaconda3\envs\nlp\Lib\site-packages\snownlp\sentiment'
    main(work_dir, dataset_path)
