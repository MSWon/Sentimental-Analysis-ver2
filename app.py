import tensorflow as tf
import preprocess
from flask import Flask, request, render_template
from model import Model

max_word_len = 100
word_dim = 512
hidden_dim = 512
num_layers = 2
max_vocab_size = 100000
corpuspath = "Movie_rating_data"
modelpath = "./model/"
modelName = "sentimental.ckpt"


def infer_example(input_sent, graph, sess):
    tokenize_words = []
    for t in prepro.twitter.pos(input_sent):
        if t[1] == "URL":
            tokenize_words.append("<URL>")
        elif t[1] == "Number":
            tokenize_words.append("<NUM>")
        else:
            tokenize_words.append('/'.join(t))
    input_idx = [prepro.sent2idx(tokenize_words)]
    input_len = [len(input_idx)]
    pred = tf.cast(tf.argmax(infer_fn, 1), tf.int32)
    score = sess.run(pred, feed_dict={graph.word_inputs: input_idx,
                                      graph.word_len: input_len})

    if (score == 1):
        return "긍정입니다"
    else:
        return "부정입니다"

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sentence = request.form["sentence"]
        result = infer_example(sentence, infer_graph, sess)
        return render_template("index.html", result=result, input_sent=sentence)
    return render_template("index.html")

if __name__ == '__main__':
    prepro = preprocess.Preprocess(word_dim=word_dim, max_vocab_size=max_vocab_size, path=corpuspath)
    word_embedding, clear_padding, word2idx, idx2word = prepro.build_embedding()
    infer_graph = Model(word_embedding, max_word_len)
    ## Create model graph
    infer_fn = infer_graph.build_model(hidden_dim, num_layers, None, False)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, modelpath + modelName)

    app.run(host='0.0.0.0', port=5000, debug=True)


