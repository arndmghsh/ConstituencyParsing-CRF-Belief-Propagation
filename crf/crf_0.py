import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from collections import defaultdict
import random, pdb
from nltk.tree import Tree
import numpy as np

tag2int = defaultdict(int)
word2int = defaultdict(int)
tag_count, word_count = 0, 0

class MyNode():
    def __init__(self, treeNode, parent):
        # treenode[0], treenode[1] are children
        # treenode._label is the label of the node
        global tag_count
        global word_count
        # Add to tag_dict
        if treeNode._label not in tag2int:
            tag2int[treeNode._label] = tag_count
            tag_count+=1
        self.true_label = np.array([tag2int[treeNode._label]])
        # Add to word dict if leaf node
        if not(isinstance(treeNode[0], Tree)):
            if treeNode[0] not in word2int:
                word2int[treeNode[0]] = word_count
                word_count+=1
            self.word = treeNode[0]

        self.true_label = torch.tensor(self.true_label).long()
        self.children = []
        self.parent = parent
        # Potentials
        self.unaryP = None
        self.edgeP = []
        # Messages from child nodes
        self.incoming_below = []
        for child in treeNode:
            if not isinstance(child, str):
                self.children.append(MyNode(child, self))


def build_tree(tree):
    my_tree = MyNode(tree, None)
    return my_tree

class Encoder(nn.Module):
    def __init__(self, word_size):
        super(Encoder, self).__init__()
        self.embedding_size = 256
        self.embedding = nn.Embedding(word_size, self.embedding_size)
        self.hidden_size = 128
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, batch_first = True)            
    def forward(self, words):
        # line: B = 1 x L x 1
        embedded_words = self.embedding(words)
        # input to LSTM = B x L x 256
        output, hidden = self.lstm(embedded_words)
        # output, (h_n, c_n) = self.lstm(embedding, (h, c)) ----- (h,c) initialized to zero
        # output size = B x L x 128
        # (h,c) are from the last time step: both have size [1,B,128]
        return output

class Linear_HtoV(nn.Module):
    def __init__(self):
        super(Linear_HtoV, self).__init__()
        self.hidden_size = 128
        self.tag_size = TAG_SIZE
        self.linear = nn.Linear(self.hidden_size, self.tag_size)
    def forward(self, input):
        return self.linear(input)

class Linear_2HtoVsq(nn.Module):
    def __init__(self):
        super(Linear_2HtoVsq, self).__init__()
        self.input_size = 128+128
        self.output_size = TAG_SIZE*TAG_SIZE
        self.linear = nn.Linear(self.input_size, self.output_size)
    def forward(self, input):
        return self.linear(input)

class Linear_HtoH(nn.Module):
    def __init__(self):
        super(Linear_HtoH, self).__init__()
        self.input_size = 128
        self.output_size = 128
        self.linear = nn.Linear(self.input_size, self.output_size)
    def forward(self, input):
        return self.linear(input)

class Linear_2HtoH(nn.Module):
    def __init__(self):
        super(Linear_2HtoH, self).__init__()
        self.input_size = 2*128
        self.output_size = 128
        self.linear = nn.Linear(self.input_size, self.output_size)
    def forward(self, input):
        return self.linear(input)

def POtraversal(node: MyNode):
    global cnt # to map the LSTM outputs to leaf nodes
    global node_count

    node_count+=1
    # Base case: if node is a leaf
    if len(node.children)==0:
        nodeH = lin_HtoH(encoder_output[0][cnt]) 
        cnt+=1
    else:
        leftH = POtraversal(node.children[0])
        if len(node.children)>1:
            rightH = POtraversal(node.children[1])
            concatH = torch.cat((leftH,rightH))
            nodeH = lin_2HtoH(concatH)
            # Left edge potential
            concatL = torch.cat((nodeH,leftH))
            edgeL = lin_2HtoVsq(concatL).view(TAG_SIZE,TAG_SIZE)
            node.edgeP.append(edgeL)
            # Right edge potential
            concatR = torch.cat((nodeH,rightH))
            edgeR = lin_2HtoVsq(concatR).view(TAG_SIZE,TAG_SIZE)
            node.edgeP.append(edgeR)
        else:
            nodeH = lin_HtoH(leftH)
            # Left edge potential
            concatL = torch.cat((nodeH,leftH))
            edgeL = lin_2HtoVsq(concatL).view(TAG_SIZE,TAG_SIZE)
            node.edgeP.append(edgeL)
    # Unary potential
    node.unaryP = lin_HtoV(nodeH)
    return nodeH

def message_LtoRoot(node: MyNode):
    # Leaf Node
    if len(node.children)==0:
        mu_NodetoParent = node.unaryP
    elif len(node.children)==1:
        # Message from left
        mu_LtoF = message_LtoRoot(node.children[0])
        a = node.edgeP[0]+mu_LtoF
        mu_LFtoNode = torch.logsumexp(a, dim=1)
        node.incoming_below.append(mu_LFtoNode)
        # Message to parent
        mu_NodetoParent = node.unaryP+mu_LFtoNode
    else:
        mu_LtoF = message_LtoRoot(node.children[0])
        # Message from left: Marginalise over L to get mu_LFtoNode
        # Edge potential shape: Parent x Child
        a = node.edgeP[0]+mu_LtoF
        # mu_LFtoNode = a.sum(dim=1)
        mu_LFtoNode = torch.logsumexp(a, dim=1)
        node.incoming_below.append(mu_LFtoNode)
        # Message from Right
        mu_RtoF = message_LtoRoot(node.children[1])
        a = node.edgeP[1]+mu_RtoF
        # mu_RFtoNode = a.sum(dim=1)
        mu_RFtoNode = torch.logsumexp(a, dim=1)
        node.incoming_below.append(mu_RFtoNode)
        # Message to parent
        mu_NodetoParent = node.unaryP+mu_LFtoNode+mu_RFtoNode
    return mu_NodetoParent

def message_RoottoL(node: MyNode, mu_from_parent):
    global ce_loss
    global n_correct
    global l_correct

    belief = node.unaryP+mu_from_parent
    for m in node.incoming_below:
        belief += m
    # Prediction
    V = belief.view(1,-1)
    ce_loss += criterion(V, torch.LongTensor([tag2int[node.true_label]]))
    # Tag prediction
    if tag2int[node.true_label] == torch.argmax(V, dim=1):
        n_correct +=1
        if len(node.children)==0:
            l_correct+=1

    for i,child in enumerate(node.children):
        mu_to_F = belief - node.incoming_below[i]
        # mu_to_F = belief
        a = (node.edgeP[i].T)+mu_to_F
        # mu_FtoCh = mu_FtoCh.sum(dim=1)
        mu_FtoCh = torch.logsumexp(a, dim=1)
        message_RoottoL(child, mu_FtoCh)
    return None


# ---------------------------------------------------------------------------------------
with open("ptb-munged.mrg.bin.oneline.txt") as f:
    lines = [l.strip() for l in f.readlines()]
    lines = lines[0:10]
    trees = [Tree.fromstring(tr1) for tr1 in lines]
    my_trees = [build_tree(tree) for tree in trees]
    sentences = [tree.leaves() for tree in trees]

EPOCHS = 10
WORD_SIZE = len(word2int)
TAG_SIZE = len(tag2int)
L_RATE = 0.0001

# Encoded words
encoded_sentences = []
for sent in sentences:
    a = torch.zeros(1,len(sent)).long()
    for i,word in enumerate(sent):
        a[0][i] = word2int[word]
    encoded_sentences.append(a)

# Encoder objects
encoder = Encoder(WORD_SIZE)
lin_HtoV = Linear_HtoV()
lin_HtoH = Linear_HtoH()
lin_2HtoH = Linear_2HtoH()
lin_2HtoVsq = Linear_2HtoVsq()
# Optimizers
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=L_RATE)
lin_HtoV_optimizer = torch.optim.Adam(lin_HtoV.parameters(), lr=L_RATE)
lin_HtoH_optimizer = torch.optim.Adam(lin_HtoH.parameters(), lr=L_RATE)
lin_2HtoH_optimizer = torch.optim.Adam(lin_2HtoH.parameters(), lr=L_RATE)
lin_2HtoVsq_optimizer = torch.optim.Adam(lin_2HtoVsq.parameters(), lr=L_RATE)
# Loss criterion
criterion = nn.CrossEntropyLoss()

# Train, Test set creation
train_sentences = encoded_sentences[0:5]
test_sentences = encoded_sentences[5:10]
train_trees = my_trees[0:5]
test_trees = my_trees[5:10]
num_train = len(train_sentences)
num_test = len(test_sentences)

train_celoss = []
train_acc = []
train_lacc = []

test_celoss = []
test_acc = []
test_lacc = []
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    epoch_acc = 0.0
    epoch_lacc = 0.0
    epoch_test_loss = 0.0
    epoch_test_acc = 0.0
    epoch_test_lacc = 0.0
    for i, sent in enumerate(train_sentences):
        encoder_optimizer.zero_grad()
        lin_2HtoH_optimizer.zero_grad()
        lin_HtoH_optimizer.zero_grad()
        lin_HtoV_optimizer.zero_grad()
        lin_2HtoVsq_optimizer.zero_grad()

        encoder_output = encoder(sent)
        cnt = 0 # To map the LSTM outputs to Leaf nodes
        ce_loss, n_correct, l_correct, node_count = 0, 0, 0, 0

        _ = POtraversal(train_trees[i])
        _ = message_LtoRoot(train_trees[i])
        message_RoottoL(train_trees[i],0)
        
        ce_loss = ce_loss/node_count
        ce_loss.backward(retain_graph=True)

        encoder_optimizer.step()
        lin_HtoV_optimizer.step()
        lin_HtoH_optimizer.step()
        lin_2HtoH_optimizer.step()
        lin_2HtoVsq_optimizer.step()

        epoch_loss += ce_loss.item()
        epoch_acc += n_correct/node_count
        epoch_lacc += l_correct/cnt
        # if i%1 == 0: print("Train Sentence: {}".format(i))

    epoch_loss = epoch_loss/num_train
    epoch_acc = epoch_acc/num_train
    epoch_lacc = epoch_lacc/num_train
    train_celoss.append(epoch_loss)
    train_acc.append(epoch_acc)
    train_lacc.append(epoch_lacc)

    # Testing ------------------------------------------------------------
    for i, sent in enumerate(test_sentences):
        encoder_output = encoder(sent)
        cnt = 0 
        ce_loss, n_correct, l_correct, node_count = 0, 0, 0, 0
        _ = POtraversal(test_trees[i])
        _ = message_LtoRoot(test_trees[i])
        message_RoottoL(test_trees[i],0)
        ce_loss = ce_loss/node_count

        epoch_test_loss += ce_loss.item()
        epoch_test_acc += n_correct/node_count
        epoch_test_lacc += l_correct/cnt
        # if i%1 == 0: print("Test Sentence: {}".format(i))

    epoch_test_loss = epoch_test_loss/num_test
    epoch_test_acc = epoch_test_acc/num_test
    epoch_test_lacc = epoch_test_lacc/num_test
    test_celoss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)
    test_lacc.append(epoch_test_lacc)

    print("Epoch: {} Train Loss: {:.2f} Train Acc: {:.2f} Train L-ACC: {:.2f}".format(
                    epoch, epoch_loss, epoch_acc, epoch_lacc))
    print("Epoch: {} Test Loss: {:.2f} Test Acc: {:.2f} Test L-ACC: {:.2f}".format(
                    epoch, epoch_test_loss, epoch_test_acc, epoch_test_lacc))
# ---------------------------------------------------------------------------------------------------
print(train_celoss)
print(train_acc)
print(train_lacc)
print('\n')
print(test_celoss)
print(test_acc)
print(test_lacc)
