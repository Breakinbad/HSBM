import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from aggregators import MeanAggregator,ConcatAggregator 


class polyHype(nn.Module): 
    def __init__(self,args,n_types, params_neighbor):
        super(polyHype, self).__init__()
        self._parse_args(args, n_types, params_neighbor)
        self._build_model()


    def _parse_args(self, args, n_types, params_neighbor):
        self.n_types = n_types
        self.use_gpu = args.cuda

        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.hidden_dim = args.dim
        self.feature_type = args.feature_type
        self.hedge_size = args.hedge_size
        self.context_hops = args.context_hops

        self.neighborhedges = torch.LongTensor(params_neighbor[0]).cuda() if args.cuda \
                else torch.LongTensor(params_neighbor[0])
        self.hyperedges = torch.LongTensor(params_neighbor[1]).cuda() if args.cuda  \
                else torch.LongTensor(params_neighbor[1])
        #print(self.hyperedges,"** hedges")
        self.hedgetypes = torch.LongTensor(params_neighbor[2]).cuda() if args.cuda else \
                torch.LongTensor(params_neighbor[2])
        self.neighbor_samples = args.neighbor_samples
        self.neighbor_agg = MeanAggregator
        print(self.neighborhedges.shape)
        print(self.hyperedges.shape)
        print(self.hedgetypes.shape)
        print(self.n_types)
        #self.neighbor_agg = ConcatAggregator


    def _build_model(self):
        
        self._build_type_feature()

        self.scores = 0.0


        self.aggregators = nn.ModuleList(self._get_neighbor_aggregators())

    def forward(self, batch):
        self.node_pairs = batch['neighbors']
        self.train_hyperedge = batch['train_hedges']
        self.labels = batch['labels']

        self._call_model()

    def _call_model(self):
        self.scores = 0.
        #for i in range(2):
        hedge_list, mask_list = self._get_neighbors_and_masks(self.labels, self.node_pairs, self.train_hyperedge)
        self.aggregated_neighbors = self._aggregate_neighbors(hedge_list, mask_list)
        #print(self.aggregated_neighbors.shape)
        self.scores += self.aggregated_neighbors
        #print(self.scores)
        self.scores_normalized = torch.sigmoid(self.scores)

    def _build_type_feature(self):
        self.type_dim = self.n_types
        self.type_features = torch.eye(self.n_types).cuda if self.use_gpu \
                else torch.eye(self.n_types)
        
        #self.type_features = torch.cat([self.type_features,
        #torch.randn([self.type_dim,1]).cuda() if self.use_gpu \
        #            else torch.randn([self.type_dim,1])], dim=-1)
       # print(self.type_features.shape,self.type_features)
        

    def _get_neighbors_and_masks(self, types, node_pairs, train_hyperedges):
        
        hedge_list = [types]
        masks = []
        train_hyperedges = torch.unsqueeze(train_hyperedges, -1) # this is the edge that we are training

        for i in range(self.context_hops):
            if i == 0:
                neighbor_nodes = node_pairs
                #print(neighbor_nodes,"nmeighbor nbdesd i111")
            else:
                neighbor_nodes = torch.index_select(self.hyperedges, 0, 
                            hedge_list[-1].view(-1)).view([self.batch_size, -1])
                #neighbor_nodes = torch.unique(neighbor_node).view([self.batch_size, -1])

        # pickes neighbors (a hyperedges i.e. list of nodes) of nodes in hyperedge: train_hyperedges
            neighbor_edges = torch.index_select(self.neighborhedges, 0,neighbor_nodes.view(-1)).view([self.batch_size,-1]) 
        # pickes neighbors of nodes in hyperedge: train_hyperedges
            hedge_list.append(neighbor_edges)
            mask = neighbor_edges - train_hyperedges
            mask = (mask != 0).float()
            
            #print("mask",mask.shape)
            masks.append(mask)
        #print(masks)
        return hedge_list, masks

    def _get_neighbor_aggregators(self):
        aggregators = []
        if self.context_hops == 1:
            aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                 input_dim=self.type_dim,
                                                 output_dim=self.n_types,
                                                 self_included=False))
        else:
            # the first layer
            aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                 input_dim=self.type_dim,
                                                 output_dim=self.hidden_dim,
                                                 act=F.relu))
            # middle layers
            for i in range(self.context_hops - 2):
                aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                     input_dim=self.hidden_dim,
                                                     output_dim=self.hidden_dim,
                                                     act=F.relu))
            # the last layer
            aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                 input_dim=self.hidden_dim,
                                                 output_dim=self.n_types,
                                                 self_included=False))
        return aggregators

    def _aggregate_neighbors(self, hedge_list, mask_list):
        edge_vectors = [torch.index_select(self.type_features,0,hedge_list[0])]
        for edges in hedge_list[1:]:
            #print(edges.shape)
            types = torch.index_select(self.hedgetypes,0,edges.view(-1)).view(list(edges.shape)+[-1])
            edge_vectors.append(torch.index_select(self.type_features,0,
                types.view(-1)).view(list(types.shape)+[-1]))
            x = torch.index_select(self.type_features,0,
                types.view(-1))
            
        for i in range(self.context_hops):
            aggregator = self.aggregators[i]
            hedge_vectors_next_iter = []
            neighbors_shape = [self.batch_size,-1,self.hedge_size,self.neighbor_samples, aggregator.input_dim]
            masks_shape = [self.batch_size,-1,self.hedge_size,self.neighbor_samples,1]
        # 1 or 2?
            for hop in range(self.context_hops - i):
                vector = aggregator(self_vectors=edge_vectors[hop], \
                        neighbor_vectors=edge_vectors[hop+1].view(neighbors_shape),
                            masks=mask_list[hop].view(masks_shape))
                hedge_vectors_next_iter.append(vector)
            edge_vectors=hedge_vectors_next_iter
        res = edge_vectors[0].view([self.batch_size, self.n_types])
        return res

    @staticmethod
    def train_step(model, optimizer, batch):
        model.train()
        optimizer.zero_grad()
        model(batch)
        criterion = nn.CrossEntropyLoss()
        loss = torch.mean(criterion(model.scores,model.labels))
        loss.backward()
        optimizer.step()

        return loss.item()

    @staticmethod
    def test_step(model,batch):
        model.eval()
        with torch.no_grad():
            model(batch)
            #print(batch,model.scores)
            acc = (model.labels == model.scores.argmax(dim=1)).float().tolist()
            #print(model.scores.argmax(dim=1).tolist())
            y_true = model.labels.tolist()
            y_pred = model.scores.argmax(dim=1).tolist()
            #auc = roc_auc_score(model.labels.tolist(),model.scores.tolist())
            emb = model.scores.tolist()
            #print(len(emb))

        return acc, model.scores_normalized.tolist(),y_true,y_pred,emb

    
    
    




