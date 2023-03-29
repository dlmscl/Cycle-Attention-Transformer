import torch
import torch.nn as nn
import math
from sparsemax import Sparsemax

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads,opt):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.use_sparsemax = opt.use_sparsemax

        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc = nn.Linear(self.head_dim * heads, embed_size, bias=False)
        

    def forward(self, input_v, input_q, input_k):
        # get the batch_size
        N = input_q.shape[0]

        # get the seq_len
        values_len, queries_len, keys_len = input_v.shape[1], input_q.shape[1], input_k.shape[1]

        # MultiHeadSelfAttention
        values = self.W_V(input_v)
        queries = self.W_Q(input_q)
        keys = self.W_K(input_k)

        # split the embedding into self.heads pieces
        values = values.reshape(N, values_len, self.heads, self.head_dim)
        queries = queries.reshape(N, queries_len, self.heads, self.head_dim)
        keys = keys.reshape(N, keys_len, self.heads, self.head_dim)

        # Einsum does matrix mult.
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if self.use_sparsemax:
            sparsemax=Sparsemax(dim=-1)
            attention =sparsemax(energy / (self.embed_size ** (1 / 2)))
        else:
            attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, queries_len, self.heads * self.head_dim)

        out = self.fc(out)

        return out

class BicrossAttention(nn.Module):
    def __init__(self, embed_size, heads,opt):
        super(BicrossAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.use_sparsemax = opt.use_sparsemax

        self.W_Q1 = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K1 = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_V1 = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc1 = nn.Linear(self.head_dim * heads, embed_size, bias=False)

        self.W_Q2 = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K2 = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_V2 = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc2 = nn.Linear(self.head_dim * heads, embed_size, bias=False)

    def forward(self, input_tensor1, input_tensor2):
        # get the batch_size
        N = input_tensor1.shape[0]

        # get the seq_len
        keys_len1, values_len1, queries_len1 = input_tensor1.shape[1], input_tensor1.shape[1], input_tensor1.shape[1]
        keys_len2, values_len2, queries_len2 = input_tensor2.shape[1], input_tensor2.shape[1], input_tensor2.shape[1]

        # BicrossAttention
        values1 = self.W_V1(input_tensor1)
        queries1 = self.W_Q1(input_tensor1)
        keys1 = self.W_K1(input_tensor1)

        values2 = self.W_V2(input_tensor2)
        queries2 = self.W_Q2(input_tensor2)
        keys2 = self.W_K2(input_tensor2)

        # split the embedding into self.heads pieces
        values1 = values1.reshape(N, values_len1, self.heads, self.head_dim)
        queries1 = queries1.reshape(N, queries_len1, self.heads, self.head_dim)
        keys1 = keys1.reshape(N, keys_len1, self.heads, self.head_dim)

        values2 = values2.reshape(N, values_len2, self.heads, self.head_dim)
        queries2 = queries2.reshape(N, queries_len2, self.heads, self.head_dim)
        keys2 = keys2.reshape(N, keys_len2, self.heads, self.head_dim)

        # Einsum does matrix mult.
        energy1 = torch.einsum("nqhd,nkhd->nhqk", [queries2, keys1])
        energy2 = torch.einsum("nqhd,nkhd->nhqk", [queries1, keys2])
        
        if self.use_sparsemax:
            sparsemax=Sparsemax(dim=3)
            attention1 =sparsemax(energy1 / (self.embed_size ** (1 / 2)))
            attention2 =sparsemax(energy2 / (self.embed_size ** (1 / 2)))
        else:
            attention1 = torch.softmax(energy1 / (self.embed_size ** (1 / 2)), dim=3)
            attention2 = torch.softmax(energy2 / (self.embed_size ** (1 / 2)), dim=3)
       
        # attention shape:(N,heads,queries_len,keys_len)

        context1 = torch.einsum("nhql,nlhd->nqhd", [attention1, values1]).reshape(N, queries_len2, self.heads * self.head_dim)
        context2 = torch.einsum("nhql,nlhd->nqhd", [attention2, values2]).reshape(N, queries_len1, self.heads * self.head_dim)

        out1 = self.fc1(context1)
        out2 = self.fc2(context2)

        return out1, out2

class TricrossAttention(nn.Module):
    def __init__(self,embed_size,heads):
        super(TricrossAttention,self).__init__()
        self.embed_size=embed_size
        self.heads=heads
        self.head_dim=self.embed_size//self.heads

        # whole slide image(C)
        self.W_Q1=nn.Linear(self.embed_size,self.heads*self.head_dim,bias=False)
        self.W_K1=nn.Linear(self.embed_size,self.heads*self.head_dim,bias=False)
        self.W_V1=nn.Linear(self.embed_size,self.heads*self.head_dim,bias=False)
        self.fc1=nn.Linear(self.heads*self.head_dim,self.embed_size,bias=False)

        #cell spatial graph(G)
        self.W_Q2 = nn.Linear(self.embed_size, self.heads * self.head_dim, bias=False)
        self.W_K2 = nn.Linear(self.embed_size, self.heads * self.head_dim, bias=False)
        self.W_V2 = nn.Linear(self.embed_size, self.heads * self.head_dim, bias=False)
        self.fc2 = nn.Linear(self.heads * self.head_dim, self.embed_size, bias=False)

        #genomic profile(S)
        self.W_Q3 = nn.Linear(self.embed_size, self.heads * self.head_dim, bias=False)
        self.W_K3 = nn.Linear(self.embed_size, self.heads * self.head_dim, bias=False)
        self.W_V3 = nn.Linear(self.embed_size, self.heads * self.head_dim, bias=False)
        self.fc3 = nn.Linear(self.heads * self.head_dim, self.embed_size, bias=False)

    def forward(self,input_tensor1,input_tensor2,input_tensor3):
        # get the batch_size
        N=input_tensor1.shape[0]

        # get the seq_len
        keys_len1,values_len1,queries_len1=input_tensor1.shape[1],input_tensor1.shape[1],input_tensor1.shape[1]
        keys_len2,values_len2,queries_len2=input_tensor2.shape[1],input_tensor2.shape[1],input_tensor2.shape[1]
        keys_len3,values_len3,queries_len3=input_tensor3.shape[1],input_tensor3.shape[1],input_tensor3.shape[1]

        # Tricross-Attention
        queries1=self.W_Q1(input_tensor1)
        keys1=self.W_K1(input_tensor1)
        values1=self.W_V1(input_tensor1)

        queries2=self.W_Q1(input_tensor2)
        keys2=self.W_K2(input_tensor2)
        values2=self.W_V2(input_tensor2)

        queries3=self.W_Q3(input_tensor3)
        keys3=self.W_K3(input_tensor3)
        values3=self.W_V3(input_tensor3)

        #split the embeding into self.heads pieces
        queries1=queries1.reshape(N,queries_len1,self.heads,self.head_dim)
        keys1=keys1.reshape(N,keys_len1,self.heads,self.head_dim)
        values1=values1.reshape(N,values_len1,self.heads,self.head_dim)

        queries2 = queries2.reshape(N, queries_len2, self.heads, self.head_dim)
        keys2 = keys2.reshape(N, keys_len2, self.heads, self.head_dim)
        values2 = values2.reshape(N, values_len2, self.heads, self.head_dim)

        queries3 = queries3.reshape(N, queries_len3, self.heads, self.head_dim)
        keys3 = keys3.reshape(N, keys_len3, self.heads, self.head_dim)
        values3 = values3.reshape(N, values_len3, self.heads, self.head_dim)

        # Einsum does matrix mult.
        keys23=torch.cat((keys2,keys3),dim=1)
        keys13=torch.cat((keys1,keys3),dim=1)
        keys12=torch.cat((keys1,keys2),dim=1)

        energy1=torch.einsum("nqhd,nkhd->nhqk",[queries1,keys23])
        energy2=torch.einsum("nqhd,nkhd->nhqk",[queries2,keys13])
        energy3=torch.einsum("nqhd,nkhd->nhqk",[queries3,keys12])

        attention1=torch.softmax(energy1/(self.embed_size**(1/2)),dim=3)
        attention2=torch.softmax(energy2/(self.embed_size**(1/2)),dim=3)
        attention3=torch.softmax(energy3/(self.embed_size**(1/2)),dim=3)

        values23 = torch.cat((values2, values3), dim=1)
        values13 = torch.cat((values1, values3), dim=1)
        values12 = torch.cat((values1, values2), dim=1)
        

        
        context1=torch.einsum("nhql,nlhd->nqhd",[attention1,values23]).reshape(N,queries_len1,self.heads*self.head_dim)
        context2=torch.einsum("nhql,nlhd->nqhd",[attention2,values13]).reshape(N,queries_len2,self.heads*self.head_dim)
        context3=torch.einsum("nhql,nlhd->nqhd",[attention3,values12]).reshape(N,queries_len3,self.heads*self.head_dim)
        out1=self.fc1(context1)
        out2=self.fc2(context2)
        out3=self.fc3(context3)

        return out1,out2,out3

class BiTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads,opt, dropout, forward_expansion=4, if_last=False):
        super(BiTransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads,opt)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.if_last = if_last
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)


    def forward(self, input):
        input_v, input_q, input_k = input
        attention = self.attention(input_v, input_q, input_k)
        x = self.dropout(self.norm1(attention + input_q))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(x + forward))
        if not self.if_last:
            return out, out, out
        else:
            return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads,opt, dropout, forward_expansion=4):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads,opt)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)


    def forward(self, input_q,input_k,input_v):
        attention = self.attention(input_v, input_q, input_k)
        x = self.dropout(self.norm1(attention + input_q))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(x + forward))
        return out

class TransformerEncoder(nn.Module):
    def __init__(self,embed_size,heads,layers,opt,dropout,forward_expansion=4):
        super(TransformerEncoder,self).__init__()
        self.embed_size=embed_size
        self.heads=heads
        self.forward_expansion=forward_expansion
        self.layers=nn.ModuleList([])
        for layer in range(layers):
            new_layer=TransformerBlock(embed_size,heads,opt,dropout, forward_expansion=self.forward_expansion)
            self.layers.append(new_layer)

    def forward(self,input_q,input_k,input_v):
        for layer in self.layers:
            input_q=layer(input_q,input_k,input_v)

        out=input_q
        return out

class BicrossTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, opt,dropout, forward_expansion=4):
        super(BicrossTransformerBlock, self).__init__()
        self.attention = BicrossAttention(embed_size, heads,opt)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.norm4 = nn.LayerNorm(embed_size)

        self.feed_forward1 = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.feed_forward2 = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input):
        input_tensor1, input_tensor2=input
        attention1, attention2 = self.attention(input_tensor1, input_tensor2)
        x1 = self.dropout(self.norm1(attention1 + input_tensor2))
        x2 = self.dropout(self.norm3(attention2 + input_tensor1))

        forward1 = self.feed_forward1(x1)
        forward2 = self.feed_forward2(x2)

        out1 = self.dropout(self.norm2(x1 + forward1))
        out2 = self.dropout(self.norm4(x2 + forward2))

        return out1, out2
    
    
class BicrossTransformerEncoder(nn.Module):
    def __init__(self,embed_size,heads,layers,opt,dropout,forward_expansion=4):
        super( BicrossTransformerEncoder,self).__init__()
        self.embed_size=embed_size
        self.heads=heads
        self.forward_expansion=forward_expansion
    
        self.layers=nn.Sequential()
        for layer in range(layers):
            self.layers.add_module("cross_attention_layer{index}".format(index=layer),BicrossTransformerBlock(self.embed_size,heads,opt,dropout, forward_expansion=self.forward_expansion))

    def forward(self,input_1,input_2):
        output_1,output_2=self.layers((input_1,input_2))
        
        out=torch.cat((output_1,output_2),dim=2)
        return out


class TricrossTransformerBlock(nn.Module):
    def __init__(self,embed_size,heads,dropout,forward_expansion):
        super(TricrossTransformerBlock,self).__init__()
        self.attention=TricrossAttention(embed_size,heads)
        self.norm11=nn.LayerNorm(embed_size)
        self.norm12=nn.LayerNorm(embed_size)
        self.norm21=nn.LayerNorm(embed_size)
        self.norm22=nn.LayerNorm(embed_size)
        self.norm31=nn.LayerNorm(embed_size)
        self.norm32=nn.LayerNorm(embed_size)

        self.feed_forward1=nn.Sequential(
            nn.Linear(embed_size,forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size,embed_size)
        )
        self.feed_forward2=nn.Sequential(
            nn.Linear(embed_size,forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size,embed_size)
        )
        self.feed_forward3 = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout=nn.Dropout(dropout)

    def forward(self,input):
        input_tensor1,input_tensor2,input_tensor3=input
        attention1,attention2,attention3=self.attention(input_tensor1,input_tensor2,input_tensor3)

        x1=self.dropout(self.norm11(attention1+input_tensor1))
        x2=self.dropout(self.norm21(attention2+input_tensor2))
        x3=self.dropout(self.norm31(attention3+input_tensor3))

        forward1=self.feed_forward1(x1)
        forward2=self.feed_forward2(x2)
        forward3=self.feed_forward3(x3)

        out1 = self.dropout(self.norm12(x1 + forward1))
        out2 = self.dropout(self.norm22(x2 + forward2))
        out3 = self.dropout(self.norm32(x3 + forward3))
        return out1,out2,out3

# position encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embed_new, dropout, max_len, device):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_new)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_new, 2).float() * (-math.log(10000.0) / embed_new))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device)

    def forward(self, vec):
        # vec shape:(N,seq_len,embed_new)
        pos = self.pe[:, :vec.size(1), :]
        vec = vec + pos
        return self.dropout(vec)

 # convolutional stem   
class Convstem(nn.Module):
    def __init__(self):
        super(Convstem, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 2, 2, 2)
        self.conv2  = torch.nn.Conv1d(2, 4, 2, 2)
    def forward(self,x):
        output=self.conv1(x)
        output=self.conv2(output)
        return output

    
# fusion1:Hierarchical Attention(multi-stream to one-stream)
class Bifusion_HierarchicalAttention(nn.Module):
    def __init__(self, opt, embed_size, heads, part, num, dropout=0.25, forward_expansion=4, max_len=5000):
        super(Bifusion_HierarchicalAttention, self).__init__()
        self.embed_new = embed_size // part
        self.use_conv1d=opt.use_conv1d
        
        # temporal convolutional layers
        if self.use_conv1d:
            self.proj_C=nn.Conv1d(self.embed_new,self.embed_new,kernel_size=1,padding=0,bias=False)
            self.proj_S=nn.Conv1d(self.embed_new,self.embed_new,kernel_size=1,padding=0,bias=False)
            
        self.num = num

        self.tflist1 = nn.Sequential()
        self.tflist2 = nn.Sequential()
        self.tflist3 = nn.Sequential()
        for i in range(num):
            if i != num - 1:
                self.tflist1.add_module("firsttflayer{index}".format(index=i),
                                        BiTransformerBlock(self.embed_new, heads, opt,dropout, forward_expansion))
                self.tflist2.add_module("secondtflayer{index}".format(index=i),
                                        BiTransformerBlock(self.embed_new, heads, opt,dropout, forward_expansion))
                self.tflist3.add_module("tflayer{index}".format(index=i),
                                        BiTransformerBlock(self.embed_new, heads, opt,dropout, forward_expansion))
            else:
                self.tflist1.add_module("firsttflayer{index}".format(index=i),
                                        BiTransformerBlock(self.embed_new, heads, opt,dropout, forward_expansion,
                                                         if_last=True))
                self.tflist2.add_module("secondtflayer{index}".format(index=i),
                                        BiTransformerBlock(self.embed_new, heads, opt,dropout, forward_expansion,
                                                         if_last=True))
                self.tflist3.add_module("thirdtflayer{index}".format(index=i),
                                        BiTransformerBlock(self.embed_new, heads, opt,dropout, forward_expansion,
                                                         if_last=True))
        self.part = part
        device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.PE = PositionalEncoding(self.embed_new, dropout, max_len, device)


    def forward(self, vec1, vec2):
        N = vec1.shape[0]
        # vec1 shape:(N,embed_size)-->(N,seq_len,embed_size)
        vec1 = vec1.reshape(N, self.part, self.embed_new)
        vec2 = vec2.reshape(N, self.part, self.embed_new)
        
        if self.use_conv1d:
            # input:[N, seq_len,embed_size]-->[N,embed_size,seq_len]
            input_C=vec1.transpose(1, 2)
            input_S=vec2.transpose(1, 2)

            #conv1d
            proj_C = self.proj_C(input_C)
            proj_S = self.proj_S(input_S)
            vec1 = proj_C.transpose(1,2)
            vec2 = proj_S.transpose(1, 2)
        #output:[N, seq_len,embed_size]
        
        vec1 = self.PE(vec1)
        z1 = self.tflist1((vec1, vec1, vec1))
        z2 = self.tflist2((vec2, vec2, vec2))
        z3 = torch.cat((z1, z2), dim=1)
        z = self.tflist3((z3, z3, z3))
        out = z.flatten(start_dim=1)
        return out


# fusion2:Cross-Attention
class Bifusion_CrossAttention(nn.Module):
    def __init__(self, opt, heads, num, dropout=0.25, forward_expansion=4, max_len=5000):
        super(Bifusion_CrossAttention, self).__init__()

        self.heads=heads
        self.dropout=dropout
        self.opt=opt
        self.forward_expansion=forward_expansion
        self.use_conv_stem=opt.use_conv_stem
            
        # 1.convolutional stem layers
        if self.use_conv_stem:
            self.convolutional_stem_C=Convstem()
            self.convolutional_stem_S=Convstem()
       
        # 2.positional encoding
        self.position_C = opt.position_C
        self.position_S = opt.position_S
        
        device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        if self.position_C or self.position_S:
            self.PE = PositionalEncoding(8, self.dropout, max_len, device)
        
        # 3.fusion with cross-transformer
        self.num = num
        self.tflist1 = BicrossTransformerEncoder(8, heads=self.heads,layers=self.num,opt=self.opt,dropout=self.dropout, forward_expansion=self.forward_expansion)

    def forward(self, input_C, input_S):
        N=input_C.shape[0]
        # 1.convolutional stem layers
        if self.use_conv_stem:
            input_C = input_C.reshape(N, 1, 32)
            input_S = input_S.reshape(N, 1, 32)
            input_C=self.convolutional_stem_C(input_C)
            input_S=self.convolutional_stem_S(input_S)
            #input_C:[64,4,8]
        else:
            input_C = input_C.reshape(N, 4, 8)
            input_S = input_S.reshape(N, 4, 8)
        
        # 2.positional encoding
        if self.position_C:
            input_C = self.PE(input_C)
        if self.position_S:
            input_S = self.PE(input_S)
            
        # 3.fusion with cross-transformer
        z = self.tflist1(input_C,input_S)

        out = z.flatten(start_dim=1)
        # out shape:(N,2*embed_size)
        return out


# fusion3:Cross-Attention to Concatenation
class Bifusion_CrossAttentionConcatenation(nn.Module):
    def __init__(self, opt, embed_size, heads, part, num=1, lastnum=1,dropout=0.25, forward_expansion=4, max_len=5000):
        super(Bifusion_CrossAttentionConcatenation, self).__init__()

        self.embed_new = embed_size // part
        self.use_conv1d=opt.use_conv1d
        
        # temporal convolutional layers
        if self.use_conv1d:
            self.proj_C=nn.Conv1d(self.embed_new,self.embed_new,kernel_size=1,padding=0,bias=False)
            self.proj_S=nn.Conv1d(self.embed_new,self.embed_new,kernel_size=1,padding=0,bias=False)
        
        self.num = num
        self.tflist1 = nn.Sequential()

        self.tflist3 = nn.Sequential()
        for i in range(num):
            self.tflist1.add_module("firsttflayer{index}".format(index=i),
                                        BicrossTransformerBlock(self.embed_new, heads,opt, dropout, forward_expansion))

        for i in range(lastnum):
            if i!=lastnum-1:
                self.tflist3.add_module("secondtflayer{index}".format(index=i),
                                    BiTransformerBlock(self.embed_new,heads,opt,dropout,forward_expansion))
            else:
                self.tflist3.add_module("secondtflayer{index}".format(index=i),
                                        BiTransformerBlock(self.embed_new, heads, opt,dropout, forward_expansion,if_last=True))

        self.part = part
        device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.PE = PositionalEncoding(self.embed_new, dropout, max_len, device)

    def forward(self, vec1, vec2):
        N = vec1.shape[0]
        # vec1 shape:(N,embed_size)-->(N,seq_len,embed_size)
        vec1 = vec1.reshape(N, self.part, self.embed_new)
        vec2 = vec2.reshape(N, self.part, self.embed_new)
        
        if self.use_conv1d:
            # input:[N, seq_len,embed_size]-->[N,embed_size,seq_len]
            input_C=vec1.transpose(1, 2)
            input_S=vec2.transpose(1, 2)

            #conv1d
            proj_C = self.proj_C(input_C)
            proj_S = self.proj_S(input_S)
            vec1 = proj_C.transpose(1,2)
            vec2 = proj_S.transpose(1, 2)
        #output:[N, seq_len,embed_size]

        vec1 = self.PE(vec1)
        z1,z2 = self.tflist1((vec1, vec2))
        z = torch.cat((z1, z2), dim=1)

        z3 = self.tflist3((z, z, z))
        out = z3.flatten(start_dim=1)
        # out shape:(N,2*embed_size)
        return out

#fusion 4: Tricross-Attention
class Trifusion_CrossAttention(nn.Module):
    def __init__(self,opt,embed_size,heads,part,num,dropout=0.25,forward_expansion=4,max_len=5000):
        super(Trifusion_CrossAttention,self).__init__()
        
        self.embed_new=embed_size//part
        self.use_conv1d=opt.use_conv1d
       
        self.num=num
        self.tflist=nn.Sequential()
        self.part = part
        device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

        for i in range(num):
            self.tflist.add_module("tflayer{index}".format(index=i),
                                   TricrossTransformerBlock(self.embed_new, heads, dropout, forward_expansion))
            
        self.position_C=opt.position_C
        self.position_G=opt.position_G
        self.position_S=opt.position_S
        
        # temporal convolutional layers
        if self.use_conv1d:
            self.proj_C=nn.Conv1d(self.embed_new,self.embed_new,kernel_size=1,padding=0,bias=False)
            self.proj_G=nn.Conv1d(self.embed_new,self.embed_new,kernel_size=1,padding=0,bias=False)
            self.proj_S=nn.Conv1d(self.embed_new,self.embed_new,kernel_size=1,padding=0,bias=False)
        
        if self.position_C or self.position_G or self.position_S:
            self.PE = PositionalEncoding(self.embed_new, dropout, max_len,device)

    def forward(self, vec1, vec2, vec3):
        N = vec1.shape[0]
        # vec1 shape:(N,embed_size)-->(N,seq_len,embed_size)
        vec1 = vec1.reshape(N, self.part, self.embed_new)
        vec2 = vec2.reshape(N, self.part, self.embed_new)
        vec3 = vec3.reshape(N, self.part, self.embed_new)

        if self.use_conv1d:
            # input:[N, seq_len,embed_size]-->[N,embed_size,seq_len]
            input_C=vec1.transpose(1,2)
            input_G=vec2.transpose(1,2)
            input_S=vec3.transpose(1, 2)

            #1.conv1d
            proj_C=self.proj_C(input_C)
            proj_G = self.proj_G(input_G)
            proj_S = self.proj_S(input_S)
            vec1 = proj_C.transpose(1,2)
            vec2 = proj_G.transpose(1, 2)
            vec3 = proj_S.transpose(1, 2)
            #output:[N, seq_len_new,d]

        
        if self.position_C:
            vec1 = self.PE(vec1)
        if self.position_G:
            vec2 = self.PE(vec2)
        if self.position_S:
            vec3 = self.PE(vec3)

        z1,z2,z3 = self.tflist((vec1, vec2,vec3))
        z = torch.cat((z1, z2,z3), dim=1)
        out = z.flatten(start_dim=1)
        # out shape:(N,3*embed_size)
        return out

#fusion 5:MULTmodel_old
class MULTmodel_old(nn.Module):
    def __init__(self,opt,embed_size,heads,part,Conly,Gonly,Sonly,layers,dropout=0.25,forward_expansion=4,max_len=5000):
        super(MULTmodel_old,self).__init__()
        self.embed_new = embed_size // part
        self.d=self.embed_new
        self.Conly=Conly
        self.Gonly=Gonly
        self.Sonly=Sonly
        self.heads=heads
        self.dropout=dropout
        self.layers=layers
        self.part=part
        self.forward_expansion=forward_expansion
        self.use_conv1d=opt.use_conv1d
        self.opt=opt

        self.position_C = opt.position_C
        self.position_G = opt.position_G
        self.position_S = opt.position_S
        device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

        if self.position_C or self.position_G or  self.position_S:
            self.PE = PositionalEncoding(self.embed_new, dropout, max_len,device)

        self.partial_mode=self.Conly+self.Gonly+self.Sonly

        # 1.temporal convolutional layers
        if self.use_conv1d:
            self.proj_C=nn.Conv1d(self.embed_new,d,kernel_size=1,padding=0,bias=False)
            self.proj_G=nn.Conv1d(self.embed_new,d,kernel_size=1,padding=0,bias=False)
            self.proj_S=nn.Conv1d(self.embed_new,d,kernel_size=1,padding=0,bias=False)

        # 2.fusion
        if self.Conly:
            self.trans_C_with_G=self.get_network(self_type='CG')
            self.trans_C_with_S=self.get_network(self_type='CS')
        if self.Gonly:
            self.trans_G_with_C=self.get_network(self_type='GC')
            self.trans_G_with_S = self.get_network(self_type='GS')
        if self.Sonly:
            self.trans_S_with_C=self.get_network(self_type='SC')
            self.trans_S_with_G = self.get_network(self_type='SG')

        # 3.self-attention
        self.trans_C=self.get_network(self_type='C',layers=4)
        self.trans_G = self.get_network(self_type='G', layers=4)
        self.trans_S = self.get_network(self_type='S', layers=4)

    def get_network(self,self_type='CS',layers=-1):
        if self_type in ['C','G','S']:
            embed_dim=2*self.d
        else:
            embed_dim=self.d
        return TransformerEncoder(embed_dim,heads=self.heads,layers=max(self.layers,layers),opt=self.opt,dropout=self.dropout,forward_expansion=self.forward_expansion)

    def forward(self,input_C,input_G,input_S):
        
        N = input_C.shape[0]
        # vec1 shape:(N,embed_size)-->(N,seq_len,embed_size)
        input_C = input_C.reshape(N, self.part, self.embed_new)
        input_G = input_G.reshape(N, self.part, self.embed_new)
        input_S = input_S.reshape(N, self.part, self.embed_new)
        
        if self.use_conv1d:
            # input:[N, seq_len,embed_size]-->[N,embed_size,seq_len]
            input_C=input_C.transpose(1,2)
            input_G=input_G.transpose(1,2)
            input_S=input_S.transpose(1, 2)

            #1.conv1d
            proj_C=self.proj_C(input_C)
            proj_G = self.proj_G(input_G)
            proj_S = self.proj_S(input_S)
            input_C = proj_C.transpose(1,2)
            input_G = proj_G.transpose(1, 2)
            input_S = proj_S.transpose(1, 2)
            #output:[N, seq_len_new,d]

        if self.position_C:
            input_C = self.PE(input_C)
        if self.position_G:
            input_G = self.PE(input_G)
        if self.position_S:
            input_S = self.PE(input_S)

        #2.fusion
        if self.Conly:
            #(G,S)-->C
            h_C_with_G=self.trans_C_with_G(input_C,input_G,input_G)
            h_C_with_S = self.trans_C_with_S(input_C,input_S, input_S)
            h_C2=torch.cat([h_C_with_G,h_C_with_S],dim=2)
            h_C2=self.trans_C(h_C2,h_C2,h_C2)
            last_h_C2= h_C2.flatten(start_dim=1)

        if self.Gonly:
            # (C,S)-->G
            h_G_with_C = self.trans_G_with_C(input_G, input_C, input_C)
            h_G_with_S = self.trans_G_with_S(input_G, input_S, input_S)
            h_G2 = torch.cat([h_G_with_C, h_G_with_S], dim=2)
            h_G2 = self.trans_G(h_G2, h_G2, h_G2)
            last_h_G2 = h_G2.flatten(start_dim=1)

        if self.Sonly:
            # (C,G)-->S
            h_S_with_C = self.trans_S_with_C(input_S, input_C,input_C)
            h_S_with_G = self.trans_S_with_G(input_S, input_G,input_G)
            h_S2 = torch.cat([h_S_with_C, h_S_with_G], dim=2)
            h_S2 = self.trans_S(h_S2, h_S2, h_S2)
            last_h_S2 = h_S2.flatten(start_dim=1)

        if self.partial_mode==1:
            if self.Conly:
                last_h_fusion=last_h_C2
            if self.Gonly:
                last_h_fusion=last_h_G2
            if self.Sonly:
                last_h_fusion = last_h_S2

        if self.partial_mode==2:
            if self.Conly and self.Gonly:
                last_h_fusion=torch.cat([last_h_C2,last_h_G2],dim=1)
            elif self.Conly and self.Sonly:
                last_h_fusion=torch.cat([last_h_C2,last_h_S2],dim=1)
            elif self.Gonly and self.Sonly:
                last_h_fusion=torch.cat([last_h_G2,last_h_S2],dim=1)
            else:
                print("Error")
        if self.partial_mode == 3:
            last_h_fusion = torch.cat([last_h_C2, last_h_G2, last_h_S2], dim=1)

        out =last_h_fusion.flatten(start_dim=1)
        return out


#fusion 5: MULTmodel
class MULTmodel(nn.Module):
    def __init__(self,opt,heads,dropout=0.25,forward_expansion=4,max_len=5000):
        super(MULTmodel,self).__init__()
        
        self.heads=heads
        self.dropout=dropout
        self.num_layer1=opt.Tfnum
        self.num_layer2=opt.lastnum
        self.forward_expansion=forward_expansion
        self.opt=opt
        
        # 1.convolutional stem layers
        self.convolutional_stem_C=Convstem()
        self.convolutional_stem_G=Convstem()
        self.convolutional_stem_S=Convstem()

        # 2.positional encoding
        self.position_C = opt.position_C
        self.position_G = opt.position_G
        self.position_S = opt.position_S
        
        device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        if self.position_C or self.position_G or  self.position_S:
            self.PE = PositionalEncoding(8, dropout, max_len,device)
            
        # 3.fusion   
        # (1) fusion1: counterclockwise: G->C->S->G    
        self.trans_C_with_G=self.get_network(self_type='CG',layers=self.num_layer1)
        self.trans_S_with_C=self.get_network(self_type='SC',layers=self.num_layer1)
        self.trans_G_with_S=self.get_network(self_type='GS',layers=self.num_layer1)
        
        # (2) fusion2: clockwise: G->S->C->G    
        self.trans_G_with_C=self.get_network(self_type='GC',layers=self.num_layer1)
        self.trans_S_with_G=self.get_network(self_type='SG',layers=self.num_layer1)
        self.trans_C_with_S=self.get_network(self_type='CS',layers=self.num_layer1)

        # (3) cross-attention
        self.cross_fusion=self.get_network(self_type='CGS',layers=self.num_layer2)
        
    def get_network(self,self_type='CS',layers=-1,embed_dim=8):
        if self_type in ['CGS']:
            embed_dim_new=3*embed_dim
            return BicrossTransformerEncoder(embed_dim_new,heads=self.heads,layers=self.num_layer2,opt=self.opt,dropout=self.dropout, forward_expansion=self.forward_expansion)
        else:
            embed_dim_new=embed_dim
            return TransformerEncoder(embed_dim_new,heads=self.heads,layers=self.num_layer1,opt=self.opt,dropout=self.dropout, forward_expansion=self.forward_expansion)

    def forward(self,input_C,input_G,input_S):
        # 1.convolutional stem layers
        N=input_C.shape[0]
        input_C = input_C.reshape(N, 1, 32)
        input_G = input_G.reshape(N, 1, 32)
        input_S = input_S.reshape(N, 1, 32)
        input_C=self.convolutional_stem_C(input_C)
        input_G=self.convolutional_stem_G(input_G)
        input_S=self.convolutional_stem_S(input_S)
        
        # 2.positional encoding
        if self.position_C:
            input_C = self.PE(input_C)
        if self.position_G:
            input_G = self.PE(input_G)
        if self.position_S:
            input_S = self.PE(input_S)
            
        # 3.fusion
        # (1) fusion1: counterclockwise: G->C->S->G 
        h_C_with_G=self.trans_C_with_G(input_C,input_G,input_G)
        h_S_with_C=self.trans_S_with_C(input_S,input_C,input_C)
        h_G_with_S=self.trans_G_with_S(input_G,input_S,input_S)
        h_ccw=torch.cat([h_C_with_G,h_S_with_C,h_G_with_S],dim=2)
        
        # (2) fusion2: clockwise: G->S->C->G  
        h_G_with_C=self.trans_G_with_C(input_G,input_C,input_C)
        h_S_with_G=self.trans_S_with_G(input_S,input_G,input_G)
        h_C_with_S=self.trans_C_with_S(input_C,input_S,input_S)
        h_cw=torch.cat([h_G_with_C,h_S_with_G,h_C_with_S],dim=2)
        
        # (3) cross-attention
        h_fusion=self.cross_fusion(h_ccw,h_cw)

        out =h_fusion.flatten(start_dim=1)
        return out

