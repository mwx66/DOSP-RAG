import json
import nltk
import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from nltk.translate import meteor_score, nist_score, ribes_score, gleu_score, bleu_score
from bert_score import score

nltk.download('wordnet')


class TextEvaluator:
    def __init__(self):
        self.metrics = {}
        self.scorers = {
            'BLEU': Bleu(n=4),
            'CIDEr': Cider(),
            'ROUGE': Rouge()
        }

    def load_data(self, truth_path, pred_path):
        """加载参考文本和预测文本"""
        with open(truth_path, 'r', encoding='utf-8') as f:

            self.truth_data = json.load(f)
            print(type(self.truth_data))
        with open(pred_path, 'r', encoding='utf-8') as f:
            self.pred_data = json.load(f)
            print(type(self.pred_data))

    def calculate_sentence_level_metrics(self, references, hypotheses):
        """计算句子级别的评估指标，支持多个预测描述"""
        scores = {}
        
        # 对每个预测计算分数
        hypothesis_scores = []
        for hyp in hypotheses:
            hyp_tokens = hyp.split()
            # 对每个reference计算分数并取最大值
            metric_scores = {
                'METEOR': max([meteor_score.meteor_score([ref.split()], hyp_tokens) for ref in references]),
                'RIBES': max([ribes_score.sentence_ribes([ref.split()], hyp_tokens) for ref in references]),
                'GLEU': max([gleu_score.sentence_gleu([ref.split()], hyp_tokens) for ref in references])
            }
            
            # 安全地计算NIST分数
            try:
                nist_scores = [
                    nist_score.sentence_nist([ref.split()], hyp_tokens)
                    if len(ref.split()) > 1 and len(hyp_tokens) > 1
                    else 0.0
                    for ref in references
                ]
                metric_scores['NIST'] = max(nist_scores) if nist_scores else 0.0
            except Exception:
                metric_scores['NIST'] = 0.0
            
            hypothesis_scores.append(metric_scores)
        
        # 计算所有预测的平均分数
        for metric in ['METEOR', 'RIBES', 'GLEU', 'NIST']:
            scores[metric] = np.mean([h_score[metric] for h_score in hypothesis_scores])
        
        return scores

    def calculate_corpus_level_metrics(self):
        """计算语料库级别的评估指标，支持多个预测描述"""
        # 对每个预测计算分数
        all_scores = []
        
        # 为每个预测创建一个新的预测数据字典
        for i in range(max(len(hyps) for hyps in self.pred_data.values())):
            pred_data_i = {}
            truth_data_i = {}
            
            # 只处理同时存在于truth和pred中的键
            for key, hyps in self.pred_data.items():
                if key in self.truth_data and i < len(hyps):
                    pred_data_i[key] = [hyps[i]]  # 仍然保持列表格式以兼容评分器
                    truth_data_i[key] = self.truth_data[key]  # 对应的参考数据
            
            if pred_data_i:  # 确保有数据再进行计算
                # 计算当前预测的所有指标
                scores_i = {}
                for scorer_name, scorer in self.scorers.items():
                    score_value, _ = scorer.compute_score(truth_data_i, pred_data_i)
                    
                    if scorer_name == 'BLEU':
                        for j, bleu_score in enumerate(score_value, start=1):
                            scores_i[f'BLEU-{j}'] = bleu_score
                    else:
                        scores_i[scorer_name] = score_value
                
                all_scores.append(scores_i)
        
        # 计算所有预测的平均分数
        if all_scores:  # 确保有分数再计算平均值
            for metric in all_scores[0].keys():
                self.metrics[metric] = np.mean([scores[metric] for scores in all_scores])

    def calculate_bertscore(self, references, candidates, lang="en"):
        """计算BERTScore"""
        single_references = [refs[0] for refs in references]
        P, R, F1 = score(candidates, single_references, lang=lang)
        return {
            "BERTScore-P": P.mean().item(),
            "BERTScore-R": R.mean().item(),
            "BERTScore-F1": F1.mean().item()
        }

    def evaluate(self, truth_path, pred_path):
        """执行完整的评估流程，支持多个预测描述"""
        # 加载数据
        self.load_data(truth_path, pred_path)

        # 准备数据
        sentence_scores = {metric: [] for metric in ['METEOR', 'NIST', 'RIBES', 'GLEU']}
        all_references = []
        all_candidates = []

        # 计算句子级别的指标
        for key, refs in self.truth_data.items():
            if key in self.pred_data:
                hyps = self.pred_data[key]  # 现在是多个预测的列表
                all_references.append(refs)
                all_candidates.extend(hyps)  # 添加所有预测

                # 计算基础指标
                scores = self.calculate_sentence_level_metrics(refs, hyps)
                for metric, score in scores.items():
                    sentence_scores[metric].append(score)

        # 更新指标平均值
        self.metrics.update({
            metric: np.mean(scores)
            for metric, scores in sentence_scores.items()
        })

        # 计算语料库级别的指标
        self.calculate_corpus_level_metrics()

        # 计算BERTScore
        bert_scores = []
        for i in range(max(len(hyps) for hyps in self.pred_data.values())):
            candidates_i = []
            references_i = []
            for key, refs in self.truth_data.items():
                if key in self.pred_data and i < len(self.pred_data[key]):
                    candidates_i.append(self.pred_data[key][i])
                    references_i.append(refs)
            
            if candidates_i:
                scores = self.calculate_bertscore(references_i, candidates_i)
                bert_scores.append(scores)
        
        # 计算BERTScore的平均值
        if bert_scores:
            avg_bert_scores = {
                metric: np.mean([scores[metric] for scores in bert_scores])
                for metric in bert_scores[0].keys()
            }
            self.metrics.update(avg_bert_scores)

        return self.metrics

    def print_results(self):
        """打印评估结果"""
        print("\n=== Caption评估指标 ===")
        for metric in sorted(self.metrics.keys()):
            print(f"{metric}: {self.metrics[metric]:.4f}")


if __name__ == "__main__":
    evaluator = TextEvaluator()
    metrics = evaluator.evaluate('data/gen_e/goldnews_en.json', 'data/gen_e/doubaow_en1.json')
    evaluator.print_results()
    
    
    

    
