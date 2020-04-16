import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):        
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Embedding vector
        self.embed = nn.Embedding(vocab_size, embed_size)        
        
        # Long Short Term Memory
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        
        # Linear Model
        self.linear_fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        """
        :param features: image features from encoder having shape (batch_size embed_size)
        :param captions: pytorch tensor corresponding to last batch of captions having shape (batch_size caption_length)
        """
        
        # Remove <end> word from tokens and get embeddings
        cap_embed = self.embed(captions[:, :-1]) 
        # print(cap_embed)
        
        # Concatenate feature vector with captions
        embeddings = torch.cat((features.unsqueeze(1), cap_embed), dim=1)  
        # print(embeddings)
        
        # Passing through LSTM Layer
        outputs, _ = self.lstm(embeddings)
        # print(outputs)
        
        # Passing through Linear Layer
        scores = self.linear_fc(outputs)
        # print(scores)
        return scores

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        preds = []
        output_length = 0
        
        while (output_length != max_len+1):
            
            # Predict output
            output, states = self.lstm(inputs, states)
            output = self.linear_fc(output.squeeze(dim = 1))
            
            # Get max value
            _, predicted_index = torch.max(output, 1)
            
            # Append word
            preds.append(predicted_index.cpu().numpy()[0].item())
            
            if (predicted_index == 1):
                break
            
            # Next input is current prediction
            inputs = self.embed(predicted_index)
            inputs = inputs.unsqueeze(1)
            
            output_length += 1
        
        return preds