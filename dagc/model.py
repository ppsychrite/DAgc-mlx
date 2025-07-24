import mlx.core as mx 
import mlx.nn as nn 

from mlx_graphs.nn.conv.gat_conv import GATConv

class DACG(nn.Module):
    def __init__(self, encoder, out_channels, chunk_size = 128, hidden_size = 200):
        super().__init__()

        self.encoder = encoder 
        self.hidden_size = hidden_size 
        self.chunk_size = chunk_size 
        self.out_channels = out_channels
        ''' 
            1. Embeddings
            2. GRU  
            3. GAT 
            4. Dense Layer 
        '''

        self.gru = nn.GRU(
            encoder.config.hidden_size,
            self.hidden_size # Appendix A lists 200 as default 
        )
        self.gat = GATConv(self.hidden_size, self.hidden_size)
        self.dense = nn.Linear(self.hidden_size * 2, out_channels)

    def __call__(self, embeddings : mx.array, speakers: list): 
        assert(embeddings.shape[0] == len(speakers)) 
        if embeddings.shape[0] > self.chunk_size: 
            embeddings = embeddings[: self.chunk_size, :]
            speakers = speakers[: self.chunk_size]
        # Just in case not full chunk length 
        rows = embeddings.shape[0]

        unique_speakers = list(set(speakers))


        # Generate utterance-embedding nodes  
        nodes = self.gru(embeddings)
        # Add speaker-embedding nodes
        glorot_init = nn.init.glorot_uniform() 
        speaker_embeddings = glorot_init(mx.zeros((len(unique_speakers), self.hidden_size)))

        nodes = mx.concatenate([nodes, speaker_embeddings])
        # Generate Adjacency list 
        adj_list = [] 
        for i, speaker in enumerate(unique_speakers): 
            node_index = rows + i  
            utterances = [i for i, s in enumerate(speakers) if s == speaker]
            for u_i in utterances: 
                adj_list.append([node_index, u_i])

        # Speaker Indices for concatenation stage 
        indices = mx.array([unique_speakers.index(s) for s in speakers])

        # is in (2, n)
        adj_list = mx.array(adj_list).transpose()

        # Only get the speaker embeddings from this 
        speaker_nodes = self.gat(adj_list, nodes)[rows :, :]
        speaker_embeddings =  speaker_nodes[indices]
        nodes = nodes[: rows, :]

        # Speaker-enriched embedings 
        embeddings = mx.concatenate([nodes, speaker_embeddings], axis = 1)

        cls = self.dense(embeddings)
        return cls 

