from transformers import BertModel, BertPreTrainedModel, DistilBertModel, DistilBertPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel
from torch import nn
from torch.nn import CrossEntropyLoss

class MTLBertForTokenClassification(BertPreTrainedModel):

    def __init__(self, config, finetune, list_labels=[], use_bilstms=False):
        #config2 = config
        #config2.num_labels = 2
        super(MTLBertForTokenClassification, self).__init__(config)
        self.num_labels = list_labels
        self.num_tasks = len(self.num_labels)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob) 
        self.use_bilstms=use_bilstms
        self.lstm_size = 400
        self.lstm_layers = 2
        self.bidirectional_lstm = True
        
        if self.use_bilstms:
            self.lstm = nn.LSTM(config.hidden_size, self.lstm_size, num_layers=self.lstm_layers, batch_first=True, 
                                bidirectional=self.bidirectional_lstm)
            
            self.hidden2tagList = nn.ModuleList([nn.Linear(self.lstm_size*(2 if self.bidirectional_lstm else 1), 
                                                       self.num_labels[idtask])
                                                       for idtask in range(self.num_tasks)])
        else:
            
            self.hidden2tagList = nn.ModuleList([nn.Linear(config.hidden_size, 
                                                       self.num_labels[idtask])
                                                       for idtask in range(self.num_tasks)])

        self.finetune = finetune
        self.init_weights()

    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        
        hidden_outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            head_mask=head_mask)

        sequence_output = hidden_outputs[0]
        
        if not self.finetune:
            sequence_output = sequence_output.detach()        

        if self.use_bilstms:
            self.lstm.flatten_parameters()
            sequence_output, hidden = self.lstm(sequence_output, None)
        
        sequence_output = self.dropout(sequence_output)
        outputs = [(classifier(sequence_output),) for classifier in self.hidden2tagList]
        losses = []   
        
        for idtask,out in enumerate(outputs):
            
            logits = out[0]
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels[idtask])[active_loss]
                    active_labels = labels[:,idtask,:].reshape(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels[idtask]), labels.view(-1))
                
                losses.append(loss)  
                 
                outputs = (sum(losses),) + hidden_outputs

        return outputs
    



class MTLRobertaForTokenClassification(RobertaPreTrainedModel):

    def __init__(self, config, finetune, list_labels=[], use_bilstms=False):
        #config2 = config
        #config2.num_labels = 2
        super(MTLRobertaForTokenClassification, self).__init__(config)
        self.num_labels = list_labels
        self.num_tasks = len(self.num_labels)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob) 
        self.use_bilstms=use_bilstms
        self.lstm_size = 400
        self.lstm_layers = 2
        self.bidirectional_lstm = True
        
        if self.use_bilstms:
            self.lstm = nn.LSTM(config.hidden_size, self.lstm_size, num_layers=self.lstm_layers, batch_first=True, 
                                bidirectional=self.bidirectional_lstm)
            
            self.hidden2tagList = nn.ModuleList([nn.Linear(self.lstm_size*(2 if self.bidirectional_lstm else 1), 
                                                       self.num_labels[idtask])
                                                       for idtask in range(self.num_tasks)])
        else:
            
            self.hidden2tagList = nn.ModuleList([nn.Linear(config.hidden_size, 
                                                       self.num_labels[idtask])
                                                       for idtask in range(self.num_tasks)])

        self.finetune = finetune
        self.init_weights()

    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        
        hidden_outputs = self.roberta(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            head_mask=head_mask)

        sequence_output = hidden_outputs[0]
        
        if not self.finetune:
            sequence_output = sequence_output.detach()        

        if self.use_bilstms:
            self.lstm.flatten_parameters()
            sequence_output, hidden = self.lstm(sequence_output, None)
        
        sequence_output = self.dropout(sequence_output)
        outputs = [(classifier(sequence_output),) for classifier in self.hidden2tagList]
        losses = []   
        
        for idtask,out in enumerate(outputs):
            
            logits = out[0]
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels[idtask])[active_loss]
                    active_labels = labels[:,idtask,:].reshape(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels[idtask]), labels.view(-1))
                
                losses.append(loss)  
                 
                outputs = (sum(losses),) + hidden_outputs

        return outputs
    
class MTLDistilBertForTokenClassification(DistilBertPreTrainedModel):
    def __init__(self, config, finetune,list_labels=[], use_bilstms=False):
        super(MTLDistilBertForTokenClassification, self).__init__(config)
        
        self.num_labels = list_labels #config.num_labels
        self.num_tasks = len(self.num_labels)
    
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)

        self.use_bilstms=use_bilstms
        self.lstm_size = 400
        self.lstm_layers = 2
        self.bidirectional_lstm = True
        
        if self.use_bilstms:
            self.lstm = nn.LSTM(config.hidden_size, self.lstm_size, num_layers=self.lstm_layers, batch_first=True, 
                                bidirectional=self.bidirectional_lstm)
            
            self.hidden2tagList = nn.ModuleList([nn.Linear(self.lstm_size*(2 if self.bidirectional_lstm else 1), 
                                                       self.num_labels[idtask])
                                                       for idtask in range(self.num_tasks)])
        else:
            
            self.hidden2tagList = nn.ModuleList([nn.Linear(config.hidden_size, 
                                                       self.num_labels[idtask])
                                                       for idtask in range(self.num_tasks)])

        self.finetune = finetune
        self.init_weights()



    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        hidden_outputs = self.distilbert(input_ids,
                            attention_mask=attention_mask,
                            head_mask=head_mask)

        sequence_output = hidden_outputs[0]
        
        if not self.finetune:
            sequence_output = sequence_output.detach()        

        if self.use_bilstms:
            self.lstm.flatten_parameters()
            sequence_output, hidden = self.lstm(sequence_output, None)
        
        sequence_output = self.dropout(sequence_output)
        outputs = [(classifier(sequence_output),) for classifier in self.hidden2tagList]
        losses = []   
        
        for idtask,out in enumerate(outputs):
            
            logits = out[0]
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels[idtask])[active_loss]
                    active_labels = labels[:,idtask,:].reshape(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels[idtask]), labels.view(-1))
                
                losses.append(loss)  
                 
                outputs = (sum(losses),) + hidden_outputs

        return outputs
