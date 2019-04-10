##-------------------Inputs------------------------
sequence_input_b = Input(shape=(seq, features), dtype='float32', name='sequence_input_a')
sequence_input_a = Input(shape=(seq, features), dtype='float32', name='sequence_input_b')
x_to_be_predicted = Input(shape=(features,), dtype='float32', name='x_to_be_predicted')
sequence_input_stat_a = Input(shape=(features_stat,), dtype='float32', name='sequence_input_stat_a')
sequence_input_stat_b = Input(shape=(features_stat,), dtype='float32', name='sequence_input_stat_b')

embedding_layer_w = Embedding(7,30)
embedding_layer_h = Embedding(24,30)

inputs_w_b = Input(shape=(16,), dtype='int32', name='input_w')
inputs_h_b = Input(shape=(16,), dtype='int32', name='input_h')

inputs_w_a = Input(shape=(16,), dtype='int32', name='input_w_a')
inputs_h_a = Input(shape=(16,), dtype='int32', name='input_h_a')

inputs_w_x = Input(shape=(1,), dtype='int32', name='input_w_x')
inputs_h_x = Input(shape=(1,), dtype='int32', name='input_h_x')

##------------------End Inputs------------------------

##-----------------Embeddings-------------------------

embed_w_b = embedding_layer_w(inputs_w_b)
embed_w_r_b = Reshape((16,30))(embed_w_b)
embed_w_a = embedding_layer_w(inputs_w_a)
embed_w_r_a = Reshape((16,30))(embed_w_a)

embed_h_b = embedding_layer_h(inputs_h_b)
embed_h_r_b = Reshape((16,30))(embed_h_b)
embed_h_a = embedding_layer_h(inputs_h_a)
embed_h_r_a = Reshape((16,30))(embed_h_a)

embed_w_x = embedding_layer_w(inputs_w_x)
embed_h_x = embedding_layer_h(inputs_h_x)

##----------------End Embeddings-----------------------

concat_b = concatenate([sequence_input_b,embed_w_r_b,embed_h_r_b],name='concat_b')
concat_a = concatenate([sequence_input_a,embed_w_r_a,embed_h_r_a], name='concat_a')
concat_x = concatenate([embed_w_x,embed_h_x], name='concat_x')

lstm_b = LSTM(1, input_shape=(seq,features), activation='relu', return_sequences=True, name='lstm_before')(concat_b)
lstm_a = LSTM(1, input_shape=(seq,features), activation='relu',go_backwards=True, return_sequences=True, name='lstm_after')(concat_a)

## lstm_b ou lstm_a = (batch,16,1)

##Attention levando em consideração os dados da lstm_b
flat_b = Flatten()(lstm_b)
attention_b = RepeatVector(16)(flat_b)
attention_b = Dense(16,activation='softmax')(attention_b)
lstm_att_b = Multiply()([flat_b, attention_b])
lstm_att_b = Lambda(lambda x: K.sum(x, axis=2))(lstm_att_b)

##Attention levando em consideração os dados da lstm_a
flat_a = Flatten()(lstm_a)
attention_a = RepeatVector(16)(flat_a)
attention_a = Dense(16, activation='softmax')(attention_a)
lstm_att_a = Multiply()([flat_a, attention_a])
lstm_att_a = Lambda(lambda x: K.sum(x, axis=2))(lstm_att_a)

fl_x = Flatten()(concat_x)

concat = concatenate([sent_representation_b, fl_x, x_to_be_predicted,sent_representation_a])

btn = BatchNormalization()(concat)

fc1 = Dense(64, activation='relu')(btn)

drop = Dropout(0.15)(fc1)

fc2 = Dense(32, activation='relu')(drop)

full = Dense(3, activation='softmax')(fc2)

model = Model([sequence_input_b,inputs_w_b,inputs_h_b,
               sequence_input_a,inputs_w_a,inputs_h_a,
               x_to_be_predicted,inputs_w_x,inputs_h_x,
               sequence_input_stat_b,
               sequence_input_stat_a], full)

adam = optimizers.Adam(lr=0.0007)
model.compile(loss=focal_loss(),
              optimizer='adam',
              metrics=['acc'])
model.summary()



Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_w (InputLayer)            (None, 16)           0                                            
__________________________________________________________________________________________________
input_h (InputLayer)            (None, 16)           0                                            
__________________________________________________________________________________________________
input_w_x (InputLayer)          (None, 1)            0                                            
__________________________________________________________________________________________________
input_h_x (InputLayer)          (None, 1)            0                                            
__________________________________________________________________________________________________
input_w_a (InputLayer)          (None, 16)           0                                            
__________________________________________________________________________________________________
input_h_a (InputLayer)          (None, 16)           0                                            
__________________________________________________________________________________________________
embedding_7 (Embedding)         multiple             210         input_w[0][0]                    
                                                                 input_w_a[0][0]                  
                                                                 input_w_x[0][0]                  
__________________________________________________________________________________________________
embedding_8 (Embedding)         multiple             720         input_h[0][0]                    
                                                                 input_h_a[0][0]                  
                                                                 input_h_x[0][0]                  
__________________________________________________________________________________________________
sequence_input_a (InputLayer)   (None, 16, 6)        0                                            
__________________________________________________________________________________________________
reshape_13 (Reshape)            (None, 16, 30)       0           embedding_7[0][0]                
__________________________________________________________________________________________________
reshape_15 (Reshape)            (None, 16, 30)       0           embedding_8[0][0]                
__________________________________________________________________________________________________
sequence_input_b (InputLayer)   (None, 16, 6)        0                                            
__________________________________________________________________________________________________
reshape_14 (Reshape)            (None, 16, 30)       0           embedding_7[1][0]                
__________________________________________________________________________________________________
reshape_16 (Reshape)            (None, 16, 30)       0           embedding_8[1][0]                
__________________________________________________________________________________________________
concat_b (Concatenate)          (None, 16, 66)       0           sequence_input_a[0][0]           
                                                                 reshape_13[0][0]                 
                                                                 reshape_15[0][0]                 
__________________________________________________________________________________________________
concat_a (Concatenate)          (None, 16, 66)       0           sequence_input_b[0][0]           
                                                                 reshape_14[0][0]                 
                                                                 reshape_16[0][0]                 
__________________________________________________________________________________________________
lstm_before (LSTM)              (None, 16, 1)        272         concat_b[0][0]                   
__________________________________________________________________________________________________
lstm_after (LSTM)               (None, 16, 1)        272         concat_a[0][0]                   
__________________________________________________________________________________________________
flatten_7 (Flatten)             (None, 16)           0           lstm_before[0][0]                
__________________________________________________________________________________________________
flatten_8 (Flatten)             (None, 16)           0           lstm_after[0][0]                 
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 16)           0           flatten_7[0][0]                  
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 16)           0           flatten_8[0][0]                  
__________________________________________________________________________________________________
repeat_vector_6 (RepeatVector)  (None, 16, 16)       0           activation_6[0][0]               
__________________________________________________________________________________________________
repeat_vector_7 (RepeatVector)  (None, 16, 16)       0           activation_7[0][0]               
__________________________________________________________________________________________________
multiply_5 (Multiply)           (None, 16, 16)       0           lstm_before[0][0]                
                                                                 repeat_vector_6[0][0]            
__________________________________________________________________________________________________
concat_x (Concatenate)          (None, 1, 60)        0           embedding_7[2][0]                
                                                                 embedding_8[2][0]                
__________________________________________________________________________________________________
multiply_6 (Multiply)           (None, 16, 16)       0           lstm_after[0][0]                 
                                                                 repeat_vector_7[0][0]            
__________________________________________________________________________________________________
lambda_5 (Lambda)               (None, 16)           0           multiply_5[0][0]                 
__________________________________________________________________________________________________
flatten_9 (Flatten)             (None, 60)           0           concat_x[0][0]                   
__________________________________________________________________________________________________
x_to_be_predicted (InputLayer)  (None, 6)            0                                            
__________________________________________________________________________________________________
lambda_6 (Lambda)               (None, 16)           0           multiply_6[0][0]                 
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 98)           0           lambda_5[0][0]                   
                                                                 flatten_9[0][0]                  
                                                                 x_to_be_predicted[0][0]          
                                                                 lambda_6[0][0]                   
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 98)           392         concatenate_3[0][0]              
__________________________________________________________________________________________________
dense_9 (Dense)                 (None, 64)           6336        batch_normalization_3[0][0]      
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 64)           0           dense_9[0][0]                    
__________________________________________________________________________________________________
dense_10 (Dense)                (None, 32)           2080        dropout_3[0][0]                  
__________________________________________________________________________________________________
dense_11 (Dense)                (None, 3)            99          dense_10[0][0]                   
==================================================================================================
Total params: 10,381
Trainable params: 10,185
Non-trainable params: 196