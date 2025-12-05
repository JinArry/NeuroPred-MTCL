import tensorflow as tf
from tensorflow.keras import layers, Model

class New_Model_06_Attn(Model):
    def __init__(
        self,
        hidden_dim=64,
        projection_dim=32,
        num_heads=2,
        num_classes=2,
        alpha=0.5,
        beta=0.05,
        dropout_rate=0.1,  # 添加dropout参数
        label_smoothing=0.0,  # 标签平滑系数，默认0.0（不使用）
        grad_clip_norm=None,  # 梯度裁剪：按L2范数裁剪，None表示不裁剪，通常1.0-5.0
        **kwargs
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta  = beta
        self.dropout_rate = dropout_rate
        self.label_smoothing = label_smoothing
        self.grad_clip_norm = grad_clip_norm

        # BiLSTM 编码
        self.encoder = layers.Bidirectional(layers.LSTM(hidden_dim, return_sequences=True))
        
        # 自注意力池化（MultiHeadAttention内部已有dropout）
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=hidden_dim,
            dropout=dropout_rate  # 保留注意力中的dropout
        )
        
        # 注意力加权池化层
        self.attention_pooling = layers.Dense(1, activation='tanh')
        # 池化后添加Dropout（恢复，但使用更小的dropout_rate）
        self.pooling_dropout = layers.Dropout(dropout_rate)
        
        # 分类
        self.classifier = layers.Dense(num_classes)
        self.aux_classifier = layers.Dense(num_classes)
        
        # 对比投影（添加Dropout）
        self.projector = tf.keras.Sequential([
            layers.Dense(hidden_dim*2, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(projection_dim)
        ])
        # 损失（添加标签平滑）
        self.ce = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            label_smoothing=label_smoothing
        )
        self.kldiv = tf.keras.losses.KLDivergence()
        self.acc = tf.keras.metrics.CategoricalAccuracy()

    def nt_xent_loss(self, z, temperature=0.5): #temperature=0.5
        z = tf.math.l2_normalize(z, axis=1)
        sim = tf.matmul(z, z, transpose_b=True) / temperature
        sim_max = tf.reduce_max(sim, axis=1, keepdims=True)
        logits = sim - sim_max
        labels = tf.range(tf.shape(z)[0])
        return tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        )

    def call(self, inputs, training=False):
        # BiLSTM编码
        x = self.encoder(inputs, training=training)
        
        # 做自注意力
        attn_out = self.attn(x, x, training=training)
        
        # 注意力加权池化
        attention_weights = self.attention_pooling(attn_out)  # [batch, seq_len, 1]
        attention_weights = tf.nn.softmax(attention_weights, axis=1)  # 归一化
        pooled = tf.reduce_sum(attn_out * attention_weights, axis=1)  # [batch, hidden_dim*2]
        # 池化后添加Dropout
        pooled = self.pooling_dropout(pooled, training=training)

        main_logits = self.classifier(pooled)
        aux_logits = self.aux_classifier(pooled)
        contra_vec = self.projector(pooled, training=training)

        # 缓存
        self._cache = {
            "main": main_logits,
            "aux": aux_logits,
            "contra": contra_vec
        }
        return main_logits

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            logits = self(x, training=True)
            c = self._cache
            loss_cls = self.ce(y, c["main"])
            loss_distill = self.kldiv(tf.stop_gradient(c["main"]), c["aux"])
            loss_contra = self.nt_xent_loss(c["contra"])

            total_loss = loss_cls + self.alpha*loss_distill + self.beta*loss_contra

        grads = tape.gradient(total_loss, self.trainable_variables)
        
        # 梯度裁剪：防止梯度爆炸
        if self.grad_clip_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.grad_clip_norm)
        
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.acc.update_state(y, tf.nn.softmax(c["main"]))
        return {"loss": total_loss, "accuracy": self.acc.result()}

    def test_step(self, data):
        x, y = data
        logits = self(x, training=False)
        self.acc.update_state(y, tf.nn.softmax(logits))
        loss = self.ce(y, logits)
        return {"loss": loss, "accuracy": self.acc.result()}