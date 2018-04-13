# keras stateful LSTM withGPU
run stateful RNN model of keras in multiple GPU

framework version<br />
> tensorflow: 1.5.0rc0<br />
> Keras: 2.1.5<br />
# Example<br />

<pre><code>
    from keras.utils.training_utils import multi_gpu_model
    from keras.layers import *
    from keras.models import *
    import multi_gpu_utils2 as multi_gpu_utils
    
    model = Sequential()
    model.add(LSTM(2, batch_input_shape=(32, None, 1), stateful=True))
    model.add(Dense(1))
    datax = np.random.rand(32, 10, 1)
    datay = np.random.rand(32, 1)
    
    # model = multi_gpu_model(model) <= invalid state size in original keras version
    
    model = multi_gpu_utils.multi_gpu_model(model)
    model.compile(loss='mean_squared_error', optimizer = 'sgd')
    model.fit(datax, datay, epochs=2,  batch_size=32)

</code></pre>
