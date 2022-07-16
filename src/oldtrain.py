import torch
import numpy as np
from collections import defaultdict
from model import polyHype
from sklearn.metrics import roc_auc_score
from predict import preprocc
from predict import calVar
args = None

def train(model_args, data):
    global args, model, sess
    args = model_args

    #hedge_and_l, n_types, neighbor_params,hedge_l2,neighbor_params2 = data
    hedge_and_l, n_types, neighbor_params = data
    nnodes = neighbor_params[0]
    print(len(nnodes),"number of nodes")

    train_hedge_l, valid_hedge_l, test_hedge_l = hedge_and_l
    train_size = len(train_hedge_l)
    print("train Sze",train_size)

    train_list = []
    for nodes in train_hedge_l:
        n = nodes[:len(nodes)-1]
        train_list.append(n)
    valid_list = []
    for nodes in valid_hedge_l:
        n = nodes[:len(nodes)-1]
        valid_list.append(n)
    test_list = []
    for nodes in test_hedge_l:
        n = nodes[:len(nodes)-1]
        test_list.append(n)

    train_hedges_all = torch.LongTensor(np.array(range(len(train_hedge_l)), np.int32))
    train_hedge = torch.LongTensor(np.array([[t] for t in train_list], np.int32))
    valid_hedge = torch.LongTensor(np.array([[t] for t in valid_list], np.int32))
    test_hedge = torch.LongTensor(np.array([[t] for t in test_list], np.int32))
            
    train_labels = torch.LongTensor(np.array([tuples[len(tuples)-1] for tuples in train_hedge_l], np.int32))
    #print(len(
    valid_labels = torch.LongTensor(np.array([tuples[len(tuples)-1] for tuples in valid_hedge_l], np.int32))
    test_labels = torch.LongTensor(np.array([tuples[len(tuples)-1] for tuples in test_hedge_l], np.int32))
           
    model = polyHype(args,n_types,neighbor_params)

    optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad,model.parameters()),
            lr=args.lr,
            )

    if args.cuda:
        model = model.cuda()
        train_labels = train_labels.cuda()
        valid_labels = valid_labels.cuda()
        test_labels = test_labels.cuda()

        train_hedges_all = train_hedges.cuda()
        train_hedge = train_hedge.cuda()
        valid_hedge = valid_hedge.cuda()
        test_hedge = test_hedge.cuda()

    # prepare for top-k evaluation
    true_types = defaultdict(set)
    for nodes in train_hedge_l:
        h = nodes[:len(nodes)-1]
        h = sorted(h)
        true_types[tuple(h)].add(nodes[len(nodes)-1])
    for nodes in valid_hedge_l:
        h = nodes[:len(nodes)-1]
        h = sorted(h)
        true_types[tuple(h)].add(nodes[len(nodes)-1])
    for nodes in test_hedge_l:
        h = nodes[:len(nodes)-1]
        h = sorted(h)
        true_types[tuple(h)].add(nodes[len(nodes)-1])

    best_valid_acc = 0.0
    final_res = None  # acc, mrr, mr, hit1, hit3, hit5

    print('start training ...')

    for step in range(args.epoch):

        # shuffle training data
        #neg_sample = generate_neg_sample(args,true_types,nnode,n_types,train_hedges_l, neighbor_params[2])
        index = np.arange(len(train_labels))
        np.random.shuffle(index)
        train_hedge = train_hedge[index]
        train_hedges_all = train_hedges_all[index]

        train_labels = train_labels[index]
        # training
        s = 0

        while s + args.batch_size <= len(train_labels):
            loss = model.train_step(model, optimizer, get_feed_dict(
                train_hedge, train_hedges_all, train_labels, s, s + args.batch_size))
            s += args.batch_size
        
        

        print(len(train_hedge),len(train_labels))
        train_acc, _,train_true,train_pred = evaluate(train_hedge, train_labels)
        print('train acc: %.4f' % (train_acc))
    embeddings = node_embedding(args,train_hedge,train_labels,nnodes,model)
    #embeddings = torch.FloatTensor(embeddings).cuda()
    X_train, X_test, label_x = calVar(embeddings, args, test_hedge)
    preprocc(X_train,label_x,X_test,test_labels)



def generate_neg_sample(args, true_samples, nnodes, n_train,n_types):

    nodes = [i for i in range(nnodes)]
    hedge = []
    for i in range(n_train):
        h_samp = np.random.choice(list(nodes),size = args.hedge_size)
        h_samp = sorted(h_samp)
        while h_samp in true_samples:
            h_samp = np.random.choice(list(nodes),size = args.hedge_size)
            h_samp = sorted(h_samp)
        h_sampl.append(n_types+1)
        hedge.append(h_samp)
    


def get_feed_dict(train_pairs, train_hedges, labels,start, end):
    feed_dict = {}
    #print(len(train_pairs[start:end]),start,end)
    feed_dict["neighbors"] = train_pairs[start:end]
    if train_hedges is not None:
        feed_dict["train_hedges"] = train_hedges[start:end]

    else:
            # for evaluation no edges should be masked out
        
        feed_dict["train_hedges"] = torch.LongTensor(np.array([-1] * (end - start), np.int32)).cuda() if args.cuda \
                else torch.LongTensor(np.array([-1] * (end - start), np.int32))
        #print(feed_dict["train_hedges"],len(feed_dict["train_hedges"]))


    feed_dict["labels"] = labels[start:end]

    return feed_dict

def get_feed_dict2(train_hyperedges, labels,start, end):
    feed_dict = {}

    feed_dict["hedges"] = train_hyperedges[start:end]

    feed_dict["labels"] = labels[start:end]

    return feed_dict

def evaluate(hyperedges, labels):

    acc_list = []
    scores_list = []
    y_true = []
    y_pred = []
    embedding = []


    s = 0
    while s + args.batch_size <= len(labels):
        acc, scores,y_t,y_p,emb = model.test_step(model, get_feed_dict(
            hyperedges, None, labels, s, s + args.batch_size))
        acc_list.extend(acc)
        scores_list.extend(scores)
        y_true.extend(y_t)
        y_pred.extend(y_p)
        s += args.batch_size
        embedding.extend(emb)
    #print(embedding)
    #ss = len(embedding[0])
    #Embedding = torch.FloatTensor(embedding).cuda() if args.cuda else torch.FloatTensor(embedding)
    #Embedding = Embedding.unsqueeze(-1)
    #x = torch.permute(Embedding,(0,2,1))
    #Embedding = torch.bmm(Embedding,x)
    #Embedding = Embedding.view([-1, ss*ss])
    #NodeEmb = []#torch.zeros([nsize,ss*ss],dtype=torch.float64)
    #for n in range(len(nnodes)):
    #    neighbor_hedges = nnodes[n]
    #    neighbor_hedges = torch.LongTensor(neighbor_hedges)
    #    neighbor_emb = torch.index_select(Embedding, 0,neighbor_hedges) 
    #    neighbor_emb = torch.mean(neighbor_emb,dim=-1).tolist()
    #    NodeEmb.append(neighbor_emb)
        #print(neighbor_emb)
        #print("******")
    #print(len(NodeEmb))
        #print(neighbor_emb.shape)




    #U,S,V = torch.pca_lowrank(Embedding, q=None, center=True, niter=2)
    #print(V.shape)

    return float(np.mean(acc_list)), np.array(scores_list), y_true, y_pred


def calculate_ranking_metrics(hyperedges_l, scores, true_types):
    #print(scores.shape[0])
    for i in range(scores.shape[0]):
        #1print(i)
#        head, tail, relation = triplets[i]
        nodes_l = hyperedges_l[i]
        nodes = nodes_l[:len(nodes_l)-1]
        sorted(nodes)
        t = nodes_l[len(nodes_l)-1]
        for j in true_types[tuple(nodes)] - {t}:
            scores[i, j] -= 1.0

    sorted_indices = np.argsort(-scores, axis=1)
    #print(scores.shape[0])
    relations = np.array(hyperedges_l)[0:scores.shape[0], 3]
    #print(relations)
    sorted_indices -= np.expand_dims(relations, 1)
    zero_coordinates = np.argwhere(sorted_indices == 0)
    rankings = zero_coordinates[:, 1] + 1

    mrr = float(np.mean(1 / rankings))
    mr = float(np.mean(rankings))
    hit1 = float(np.mean(rankings <= 1))
    hit3 = float(np.mean(rankings <= 3))
    hit5 = float(np.mean(rankings <= 5))

    return mrr, mr, hit1, hit3, hit5

def node_embedding(args,hyperedges,labels,nnodes,model):

    embedding = []


    s = 0
    #print(len(labels),"aaaaaaaaaaaaaaaa")
    batch_size = 1
    while s + args.batch_size <= len(labels):
        _, _,_,_,emb = model.test_step(model, get_feed_dict(
            hyperedges, None, labels, s, s + args.batch_size))
        s += args.batch_size
        embedding.extend(emb)
    #print(embedding)
    ss = len(embedding[0])
    Embedding = torch.FloatTensor(embedding).cuda() if args.cuda else torch.FloatTensor(embedding)
    Embedding = Embedding.unsqueeze(-1)
    x = torch.permute(Embedding,(0,2,1))
    Embedding = torch.bmm(Embedding,x)
    Embedding = Embedding.view([-1, ss*ss])
    print(Embedding.shape,"shape")
    #Embedding = Embedding.tolist()
    embw = open("nodeEmbedding.txt","w")
    print(len(Embedding[0]))

    NodeEmb = []#torch.zeros([nsize,ss*ss],dtype=torch.float64)
    for n in range(len(nnodes)):
        neighbor_hedges = nnodes[n]
        neighbor_hedges = torch.LongTensor(neighbor_hedges)
        #print(neighbor_hedges,n)
        neighbor_emb = torch.index_select(Embedding, 0,neighbor_hedges) 
        neighbor_emb = torch.mean(neighbor_emb,dim=-2).tolist()

        NodeEmb.append(neighbor_emb)
    f = open("iJbin.txt","r")
    i = 0
    for line in f:
        w = line.split()
        tmp = []
        for a in w:
            val = float(a)
            tmp.append(val)
        NodeEmb[i].extend(tmp)
        i += 1

    for i in range(len(NodeEmb)):
        #embw.write(str(i))
        #embw.write(" ")
        for e in NodeEmb[i]:
            embw.write(str(e))
            embw.write(" ")
        embw.write("\n")


    return NodeEmb


