import tensorflow as tf
from triplet_loss import batch_hard_triplet_loss, batch_all_triplet_loss


def build_model(images, mode, params):
    """Compute outputs of the model (embeddings).
    Args:
        images: (tf.Tensor) contains the inputs of the graph (features)
                this can be `tf.placeholder` or outputs of `tf.data`
        mode: (int) An instance of tf.estimator.ModeKeys (TRAIN, EVAL, PREDICT)
        params: (dict) experiment parameters
    Returns:
        model: (tf.Tensor) output of the model
    """

    with tf.variable_scope("model"):
        model = tf.layers.conv2d(images, 16, [3,3], strides=(2, 2), activation=tf.nn.relu)
        model = tf.layers.conv2d(model, 32, [3,3], strides=(1, 1), activation=tf.nn.relu)
        model = tf.layers.conv2d(model, 64, [3,3], strides=(2, 2), activation=tf.nn.relu)
        model = tf.reduce_mean(model, axis=[1,2])   # global avg pooling

        model = tf.layers.dense(model, params.embedding_size)   # do not add activation here

        # if using cross_entropy loss, add a FC layer to output the probability of each class
        if mode != tf.estimator.ModeKeys.PREDICT and params.loss == "cross_entropy":
            model = tf.layers.dense(model, params.num_classes)
    return model


def model_fn(features, labels, mode, params):
    """Model function for tf.estimator

    Args:
        features: (tf.Tensor) input batch of images
        labels: (tf.Tensor) labels of the images
        mode: (int) An instance of tf.estimator.ModeKeys (TRAIN, EVAL, PREDICT)
        params: (dict) experiment parameters

    Returns:
        model_spec: (tf.estimator.EstimatorSpec)
    """

    images = features
    tf.summary.image('train_image', images, max_outputs=3)

    # Create model
    embeddings = build_model(images, mode, params)

    # if predicting for new data, just compute and return the embeddings
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'embeddings': embeddings}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
   

    # Define loss
    if params.loss == "triplet_batch_hard":
        loss = batch_hard_triplet_loss(labels, embeddings, margin=params.margin)
    elif params.loss == "triplet_batch_all":
        loss = batch_all_triplet_loss(labels, embeddings, margin=params.margin, metrics=metrics)
    elif params.loss == "cross_entropy":
        one_hot_labels = tf.one_hot(labels, params.num_classes)
        loss = tf.losses.softmax_cross_entropy(one_hot_labels, embeddings)
    tf.summary.scalar('loss', loss)


    # Metrics
    metrics = {}
    embedding_mean_norm = tf.metrics.mean(tf.norm(embeddings, axis=1))
    metrics["metrics/embedding_mean_norm"] = embedding_mean_norm
    with tf.name_scope("metrics/"):
        tf.summary.scalar('embedding_mean_norm', embedding_mean_norm[1])   

    if params.loss == "cross_entropy":
        predictions = tf.argmax(embeddings, 1)
        accuracy =  tf.metrics.accuracy(labels=labels,
                                        predictions=predictions)
        metrics["metrics/accuracy"] = accuracy
        with tf.name_scope("metrics/"):     #slash / prevents scope from being unique and reenters the name scope     
            tf.summary.scalar('accuracy', accuracy[1])  # The tf.summary.scalar will make accuracy available to TensorBoard
                                                        # in both TRAIN and EVAL modes.


    if mode == tf.estimator.ModeKeys.EVAL:
        # loss and metrics run on validation set, https://www.tensorflow.org/guide/custom_estimators
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics) #Values of eval_metric_ops must be (metric_value, update_op) tuples


    # Optimizer
    optimizer = tf.train.AdamOptimizer(params.learning_rate)
    global_step = tf.train.get_global_step()


    train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks = [])