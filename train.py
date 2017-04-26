import numpy as np
import tensorflow as tf
from datetime import datetime
import mr_rnn, data_aux
import os
import sys

if __name__ == "__main__":
	# Settings
	evaluation_step_size = 2

	batch_size = 200
	num_seq = 6
	num_epochs = 100

	learning_rate = 0.002
	gradient_clipping = 1.

	n_hidden_coarse_prediction = 500
	nl_kwargs = {'num_seq' : num_seq,
		'num_steps' : 20,
		'batch_size' : batch_size,
		'n_hidden_encoder' : 500, # for backward and forward cell each
		'n_hidden_context' : 1000, 
		'n_hidden_decoder' : 2000}
	coarse_kwargs = {'num_seq' : num_seq,
		'num_steps' : nl_kwargs['num_steps'] + 1,
		'batch_size' : batch_size,
		'n_hidden_encoder' : 1000, # for backward and forward cell each
		'n_hidden_context' : 1000,
		'n_hidden_decoder' : 2000}

	# Load data and create batches
	nl_train = sys.argv[1]
	nl_valid = sys.argv[2]
	coarse_train = sys.argv[3]
	coarse_valid = sys.argv[4]
	outDir = sys.argv[5]

	timer = datetime.now()
	data = data_aux.generate_full_ubuntu_data_with_coarse(nl_train, nl_valid, coarse_train, coarse_valid, max_utterances = num_seq, max_tokens = nl_kwargs['num_steps'], min_frequency_nl = 10, min_frequency_coarse = 10)
	batches = data_aux.batch_iter([data['coarse_data_train'], data['coarse_length_train'], data['nl_data_train'], data['nl_length_train']],batch_size=batch_size, num_epochs=num_epochs)
	timer = datetime.now() - timer
	print('Data loaded, time spent:', timer)

	# Load word embedding
	timer = datetime.now()
	coarse_W_embedding = data_aux.pretrained_embedding(data['coarse_vocab_processor'])
	nl_W_embedding = data_aux.pretrained_embedding(data['nl_vocab_processor'])
	#coarse_W_embedding = data_aux.random_embedding(data['coarse_vocab_processor'], embedding_dim =300)
	#nl_W_embedding = data_aux.random_embedding(data['nl_vocab_processor'], embedding_dim =300)
	coarse_kwargs['embedding_shape'] = coarse_W_embedding.shape
	nl_kwargs['embedding_shape'] = nl_W_embedding.shape
	timer = datetime.now() - timer
	print('Embeddings loaded, time spent:', timer)

	# Build graph
	timer = datetime.now()
	tf.reset_default_graph()
	graph_nodes = mr_rnn.build_graph(coarse_kwargs, n_hidden_coarse_prediction, nl_kwargs, gradient_clipping)
	timer = datetime.now() - timer
	print('Graph built, time spent:', timer)

	restore_model = ""

	# Train network
	acc_loss = 0
	with tf.Session() as sess:
		timer = datetime.now()
		feed_dict={graph_nodes['coarse_vocab_input'] : coarse_W_embedding,
				   graph_nodes['nl_vocab_input'] : nl_W_embedding
				  }
		sess.run(tf.initialize_all_variables(), feed_dict=feed_dict)

		summary_writer = tf.train.SummaryWriter(outDir, sess.graph)
		summary = tf.Summary()
		saver = tf.train.Saver()

		timer = datetime.now() - timer
		print('Time init graph: ', timer)
		timer = datetime.now()
		acc_loss = 0
		
		if restore_model is not "":
			ckpt = tf.train.get_checkpoint_state(restore_model)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				print("Model Successfully Loaded")
		

		for i, batch in enumerate(batches):
			coarse_seq, coarse_len, nl_seq, nl_len = batch
			feed_dict = {graph_nodes['coarse_sequence_input'] : coarse_seq,
						 graph_nodes['coarse_length_input'] : coarse_len,
						 graph_nodes['nl_sequence_input'] : nl_seq,
						 graph_nodes['nl_length_input'] : nl_len,
						 graph_nodes['learning_rate'] : learning_rate
						 }
			loss, _ = sess.run([graph_nodes['total_loss'], graph_nodes['train_step']], feed_dict=feed_dict)
			print('Train step: %d, loss: %7f, time spent training: %s' % (i, loss, str(datetime.now() - timer)))
			summary = tf.Summary(value=[tf.Summary.Value(tag="training_loss", simple_value=loss.item() ),])
			summary_writer.add_summary(summary, i)

			if i % 400 == 0:
				total_loss_val = 0
				batches_val = data_aux.batch_iter([data['coarse_data_dev'],
									  data['coarse_length_dev'],
									  data['nl_data_dev'],
									  data['nl_length_dev']],
									  batch_size=batch_size,
									  num_epochs=1)
				for (j, batch_val) in enumerate(batches_val):
					(coarse_seq_val, coarse_len_val, nl_seq_val, nl_len_val) = batch_val
					feed_dict_val = {
						graph_nodes['coarse_sequence_input']: coarse_seq_val,
						graph_nodes['coarse_length_input']: coarse_len_val,
						graph_nodes['nl_sequence_input']: nl_seq_val,
						graph_nodes['nl_length_input']: nl_len_val,
						graph_nodes['learning_rate'] : 0.0
						}
					loss_val = sess.run(graph_nodes['total_loss'],feed_dict=feed_dict_val)
					total_loss_val = total_loss_val + loss_val
					print('Validation step: %d, loss: %7f, time spent training: %s' % (j, loss_val, str(datetime.now() - timer)))
				summary = tf.Summary(value=[tf.Summary.Value(tag="validation_loss", simple_value=(total_loss_val/j).item() ),])
				summary_writer.add_summary(summary, i)
				print('Validation Loss: %7f' % total_loss_val)
				checkpoint_path = outDir + 'model'+ str(i) + '.ckpt'
				saver.save(sess, checkpoint_path, global_step=i)