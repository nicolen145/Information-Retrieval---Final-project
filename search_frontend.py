from flask import Flask, request, jsonify

import re
import math
import pickle
from collections import Counter, defaultdict

from google.cloud import storage
from inverted_index_gcp import InvertedIndex


# CONFIG
BUCKET_NAME = "209296169"
BASE_DIR = "postings_gcp"
INDEX_NAME = "index"  
TITLES_BLOB = f"{BASE_DIR}/titles.pkl"

PAGERANK_CANDIDATES = [
    f"{BASE_DIR}/pagerank.pkl",
    "pagerank.pkl",
    "pr/pagerank.pkl",
]
PAGEVIEWS_CANDIDATES = [
    f"{BASE_DIR}/wid2pv.pkl",
    f"{BASE_DIR}/pageviews.pkl",
    "pageviews/wid2pv.pkl",
    "wid2pv.pkl",
]

# Number of docs 
N_DOCS_APPROX = 6_348_910

# GLOBAL RESOURCES
_STORAGE_CLIENT = None
_IDX = None
_TITLES = None
_PAGERANK = None
_PAGEVIEWS = None

def _get_storage_client():
    global _STORAGE_CLIENT
    if _STORAGE_CLIENT is None:
        _STORAGE_CLIENT = storage.Client()
    return _STORAGE_CLIENT

def _load_index_and_titles():
    """Load index + titles once per process (allowed; not query result caching)."""
    global _IDX, _TITLES
    if _IDX is None:
        _IDX = InvertedIndex.read_index(BASE_DIR, INDEX_NAME, BUCKET_NAME)
    if _TITLES is None:
        client = _get_storage_client()
        b = client.bucket(BUCKET_NAME)
        _TITLES = pickle.loads(b.blob(TITLES_BLOB).download_as_bytes())
    return _IDX, _TITLES

def _load_optional_dict(candidates):
    """Try loading a dict-like pickle from one of several GCS paths."""
    client = _get_storage_client()
    b = client.bucket(BUCKET_NAME)
    for path in candidates:
        try:
            return pickle.loads(b.blob(path).download_as_bytes())
        except Exception:
            continue
    return {}

RE_WORD = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9\-']{1,23}", re.UNICODE)
ENGLISH_STOPWORDS = frozenset({
    'a','about','above','after','again','against','ain','all','am','an','and',
    'any','are','aren',"aren't",'as','at','be','because','been','before','being',
    'below','between','both','but','by','can','couldn',"couldn't",'d','did',
    'didn',"didn't",'do','does','doesn',"doesn't",'doing','don',"don't",'down',
    'during','each','few','for','from','further','had','hadn',"hadn't",'has',
    'hasn',"hasn't",'have','haven',"haven't",'having','he',"he'd","he'll","he's",
    'her','here','hers','herself','him','himself','his','how','i',"i'd","i'll",
    "i'm","i've",'if','in','into','is','isn',"isn't",'it',"it'd","it'll","it's",
    'its','itself','just','ll','m','ma','me','mightn',"mightn't",'more','most',
    'mustn',"mustn't",'my','myself','needn',"needn't",'no','nor','not','now','o',
    'of','off','on','once','only','or','other','our','ours','ourselves','out',
    'over','own','re','s','same','shan',"shan't",'she',"she'd","she'll","she's",
    'should',"should've",'shouldn',"shouldn't",'so','some','such','t','than',
    'that',"that'll",'the','their','theirs','them','themselves','then','there',
    'these','they',"they'd","they'll","they're","they've",'this','those',
    'through','to','too','under','until','up','ve','very','was','wasn',"wasn't",
    'we',"we'd","we'll","we're","we've",'were','weren',"weren't",'what','when',
    'where','which','while','who','whom','why','will','with','won',"won't",
    'wouldn',"wouldn't",'y','you',"you'd","you'll","you're","you've",'your',
    'yours','yourself','yourselves'
})

CORPUS_STOPWORDS = {
    "category", "references", "also", "external", "links",
    "may", "first", "see", "history", "people", "one", "two",
    "part", "thumb", "including", "second", "following",
    "many", "however", "would", "became"
}

STOPWORDS = frozenset(ENGLISH_STOPWORDS | CORPUS_STOPWORDS)


def tokenize(text: str):
    if not text:
        return []
    tokens = [m.group(0).lower() for m in RE_WORD.finditer(text)]
    return [t for t in tokens if t not in STOPWORDS]



class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    idx, titles = _load_index_and_titles()

    q_tokens = tokenize(query)
    if not q_tokens:
        res = []
    else:
        q_tf = Counter(q_tokens)

        # query weights
        q_w = {}
        q_norm_sq = 0.0
        for t, tf in q_tf.items():
            df = idx.df.get(t, 0)
            if df == 0:
                continue
            idf = math.log((N_DOCS_APPROX + 1) / df)
            wq = (1.0 + math.log(tf)) * idf
            q_w[t] = wq
            q_norm_sq += wq * wq

        if not q_w:
            res = []
        else:
            q_norm = math.sqrt(q_norm_sq)

            dot = defaultdict(float)
            d_norm_sq = defaultdict(float)
            matched = defaultdict(int)

            for t, wq in q_w.items():
                df = idx.df[t]
                idf = math.log((N_DOCS_APPROX + 1) / df)
                for doc_id, tf in idx.read_a_posting_list(BASE_DIR, t, BUCKET_NAME):
                    doc_id = int(doc_id)
                    matched[doc_id] += 1
                    wd = (1.0 + math.log(tf)) * idf
                    dot[doc_id] += wd * wq
                    d_norm_sq[doc_id] += wd * wd

            # 1) candidate list from body
            candidates = []
            for doc_id, val in dot.items():
                dn = math.sqrt(d_norm_sq[doc_id])
                if dn == 0:
                    continue
                cos = val / (dn * q_norm)
                cov = matched[doc_id]
                body_score = cos * (1.0 + 0.15 * cov)
                candidates.append((doc_id, body_score))

            candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = candidates[:2000]   # rerank only top candidates

            # 2) title boost rerank
            q_terms = set(q_tokens)
            reranked = []
            for doc_id, body_score in candidates:
                title = titles.get(doc_id, "")
                if title:
                    title_terms = set(tokenize(title))
                    title_match = len(q_terms & title_terms)
                else:
                    title_match = 0

                final = body_score * (1.0 + 0.30 * title_match)
                reranked.append((doc_id, final))

            reranked.sort(key=lambda x: x[1], reverse=True)
            res = [(doc_id, titles.get(doc_id, "")) for doc_id, _ in reranked[:100]]
    # END SOLUTION

    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    idx, titles = _load_index_and_titles()

    q_tokens = tokenize(query)
    if not q_tokens:
        res = []
    else:
        q_tf = Counter(q_tokens)

        # build query weights
        q_w = {}
        q_norm_sq = 0.0
        for t, tf in q_tf.items():
            df = idx.df.get(t, 0)
            if df == 0:
                continue
            idf = math.log((N_DOCS_APPROX + 1) / df)
            wq = (1.0 + math.log(tf)) * idf
            q_w[t] = wq
            q_norm_sq += wq * wq

        if not q_w:
            res = []
        else:
            q_norm = math.sqrt(q_norm_sq)

            dot = defaultdict(float)
            d_norm_sq = defaultdict(float)
            matched = defaultdict(int)

            for t, wq in q_w.items():
                df = idx.df[t]
                idf = math.log((N_DOCS_APPROX + 1) / df)
                for doc_id, tf in idx.read_a_posting_list(BASE_DIR, t, BUCKET_NAME):
                    doc_id = int(doc_id)
                    matched[doc_id] += 1
                    wd = (1.0 + math.log(tf)) * idf
                    dot[doc_id] += wd * wq
                    d_norm_sq[doc_id] += wd * wd

            scored = []
            for doc_id, val in dot.items():
                dn = math.sqrt(d_norm_sq[doc_id])
                if dn == 0:
                    continue
                cos = val / (dn * q_norm)

                # coverage boost
                cov = matched[doc_id]
                final = cos * (1.0 + 0.15 * cov)

                scored.append((doc_id, final))

            scored.sort(key=lambda x: x[1], reverse=True)
            res = [(doc_id, titles.get(doc_id, "")) for doc_id, _ in scored[:100]]
    # END SOLUTION

    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    idx, titles = _load_index_and_titles()

    q_terms = set(tokenize(query))
    hits = defaultdict(int)

    for term in q_terms:
        if term not in idx.df:
            continue
        for doc_id, _tf in idx.read_a_posting_list(BASE_DIR, term, BUCKET_NAME):
            hits[int(doc_id)] += 1

    ranked = sorted(hits.items(), key=lambda x: x[1], reverse=True)
    res = [(doc_id, titles.get(doc_id, "")) for doc_id, _ in ranked]
    # END SOLUTION

    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    global _PAGERANK
    if _PAGERANK is None:
        _PAGERANK = _load_optional_dict(PAGERANK_CANDIDATES)

    pr = _PAGERANK
    res = [float(pr.get(int(wid), 0.0)) for wid in wiki_ids]
    # END SOLUTION

    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    global _PAGEVIEWS
    if _PAGEVIEWS is None:
        _PAGEVIEWS = _load_optional_dict(PAGEVIEWS_CANDIDATES)

    pv = _PAGEVIEWS
    res = [int(pv.get(int(wid), 0)) for wid in wiki_ids]
    # END SOLUTION

    return jsonify(res)

def run(**options):
    app.run(**options)

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
