import mlx.core as mx 
import mlx.nn as nn 

from mlx_graphs.nn.conv.gat_conv import GATConv

class DAGC(nn.Module):
    def __init__(self, encoder, out_channels, chunk_size = 128, hidden_size = 200):
        super().__init__()
        assert(hidden_size % 2 == 0)

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

        # Bidrectional GRU is just two GRUs that get concatenated at the end 
        self.forward_gru = nn.GRU(
            encoder.config.hidden_size,
            int(self.hidden_size / 2)
        )
        self.backward_gru = nn.GRU(
            encoder.config.hidden_size,
            int(self.hidden_size / 2)
        )

        # Appendix lists two layers with 2 attention heads each 
        self.gat1 = GATConv(self.hidden_size, int(self.hidden_size / 2), heads = 2)
        self.gat2 = GATConv(self.hidden_size, int(self.hidden_size/  2), heads = 2)
        self.dense = nn.Linear(self.hidden_size * 2, out_channels)

        # For the encoder, only unfreeze the final layer. 
        self.encoder.freeze()
        self.encoder.modules()[0]['model']['layers'][-1].unfreeze()


    def __call__(self, input_ids: mx.array, attention_mask: mx.array, position_ids: mx.array, speakers: list): 
        # Row check
        assert(input_ids.shape[0] == len(speakers)) 
        assert(input_ids.shape[0] == attention_mask.shape[0])
        assert(position_ids.shape[0] == attention_mask.shape[0])

        # Column check 
        assert(input_ids.shape[1] == attention_mask.shape[1])
        assert(input_ids.shape[1] == position_ids.shape[1])

        output = self.encoder(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            position_ids = position_ids
        )
        embeddings = output.last_hidden_state[:, 0, :]

        if embeddings.shape[0] > self.chunk_size: 
            embeddings = embeddings[: self.chunk_size, :]
            speakers = speakers[: self.chunk_size]
        # Just in case not full chunk length 
        rows = embeddings.shape[0]

        unique_speakers = list(set(speakers))


        # Generate utterance-embedding nodes  

        # Pass thru forwards and backwards gru and concatenate results 
        left = self.forward_gru(embeddings)        
        right = self.backward_gru(embeddings)[::-1] # Flip upside-down  


        nodes = mx.concatenate([left, right], axis = 1)


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

        speaker_nodes = self.gat1(adj_list, nodes)
        # Only get the speaker embeddings from this 
        speaker_nodes = self.gat2(adj_list, speaker_nodes)[rows :, :]
        speaker_embeddings =  speaker_nodes[indices]
        nodes = nodes[: rows, :]

        # Speaker-enriched embedings 
        embeddings = mx.concatenate([nodes, speaker_embeddings], axis = 1)

        cls = self.dense(embeddings)
        return cls 

