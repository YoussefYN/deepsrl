from nltk.tokenize import word_tokenize

from deepsrl.neural_srl.shared import *
from deepsrl.neural_srl.shared.constants import *
from deepsrl.neural_srl.shared.dictionary import Dictionary
from deepsrl.neural_srl.shared.inference import *
from deepsrl.neural_srl.shared.io_utils import bio_to_spans
from deepsrl.neural_srl.shared.reader import string_sequence_to_ids
from deepsrl.neural_srl.shared.tagger_data import TaggerData
from deepsrl.neural_srl.theano.tagger import BiLSTMTaggerModel


class DeepSrl:
    def __init__(self, srl_model_path, pid_model_path, embeddings_path):
        self._embeddings_dir = embeddings_path
        pid_model, self._pid_data = self._load_model(pid_model_path, 'propid')
        srl_model, self._srl_data = self._load_model(srl_model_path, 'srl')
        self._transition_params = get_transition_params(self._srl_data.label_dict.idx2str)

        self._pid_pred_function = pid_model.get_distribution_function()
        self._srl_pred_function = srl_model.get_distribution_function()

    def _load_model(self, model_path, model_type):
        config = configuration.get_config(os.path.join(model_path, 'config'))
        # Load word and tag dictionary
        word_dict = Dictionary(unknown_token=UNKNOWN_TOKEN)
        label_dict = Dictionary()
        word_dict.load(os.path.join(model_path, 'word_dict'))
        label_dict.load(os.path.join(model_path, 'label_dict'))
        data = TaggerData(config, [], [], word_dict, label_dict, None, None)

        if model_type == 'srl':
            test_sentences, emb_inits, emb_shapes = reader.get_srl_test_data(
                None, config, data.word_dict, data.label_dict, self._embeddings_dir, False)
        else:
            test_sentences, emb_inits, emb_shapes = reader.get_postag_test_data(
                None, config, data.word_dict, data.label_dict, self._embeddings_dir, False)

        data.embedding_shapes = emb_shapes
        data.embeddings = emb_inits
        model = BiLSTMTaggerModel(data, config=config, fast_predict=True)
        model.load(os.path.join(model_path, 'model.npz'))
        return model, data

    def sr_predict(self, sentence):
        response_arr = []
        tokenized_sent = word_tokenize(sentence)
        num_tokens = len(tokenized_sent)
        s0 = string_sequence_to_ids(tokenized_sent, self._pid_data.word_dict, True)
        l0 = [0] * len(s0)
        x, _, _, weights = self._pid_data.get_test_data([(s0, l0)])
        pid_pred, scores0 = self._pid_pred_function(x, weights)
        s1_sent = string_sequence_to_ids(tokenized_sent, self._srl_data.word_dict, True)
        s1 = []
        predicates = []
        for i, p in enumerate(pid_pred[0]):
            if self._pid_data.label_dict.idx2str[p] == 'V':
                predicates.append(i)
                feats = [1 if j == i else 0 for j in range(num_tokens)]
                s1.append((s1_sent, feats, l0))
        if len(s1) == 0:
            # If no identified predicate.
            return response_arr

        # Semantic role labeling.
        x, _, _, weights = self._srl_data.get_test_data(s1)
        srl_pred, scores = self._srl_pred_function(x, weights)
        arguments = []
        for i, sc in enumerate(scores):
            viterbi_pred, _ = viterbi_decode(sc, self._transition_params)
            arg_spans = bio_to_spans(viterbi_pred, self._srl_data.label_dict)
            arguments.append(arg_spans)
        for (pred, args) in zip(predicates, arguments):
            action_response = self._handle_predicate(args, pred, tokenized_sent)
            if action_response is not None:
                response_arr.append(action_response)
        return response_arr

    def _handle_predicate(self, args, pred, tokenized_sent):
        partial_response = {}
        for arg in args:
            partial_response[arg[0]] = " ".join(tokenized_sent[arg[1]:arg[2] + 1])
        partial_response["v_index"] = pred
        return partial_response