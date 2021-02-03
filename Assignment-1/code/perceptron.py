import numpy as np
import argparse

def train(input_vec, labels, mode_val, epochs = 100, learn_rate =0.1):
	weight = np.zeros(len(input_vec[0])+1)
	if mode_val == "erm":
		
		erm_prev = 0
		for i in range(epochs):
			erm_sum = 0
			
			for inputs, label in zip(input_vec, labels):
				predict_val = predict(inputs, weight)
				weight[1:] += learn_rate * (label - predict_val) * inputs
				weight[0] += learn_rate * (label - predict_val)
				erm_sum += abs(label - predict_val)
			erm_loss = erm_sum/len(input_vec)
			if erm_loss ==0:
				print('Converged due to zero error at epoch', i)
				break
			elif erm_loss == erm_prev:
				print('Converged due to error being constant at epoch', i)
				break
			erm_prev =erm_loss

		print("ERM Loss:", erm_loss)
		print("ERM Accuracy:", 1-erm_loss)


	elif mode_val== "crossvalidation":
		fold = []
		fold_labels = []
		extras  = len(input_vec)%10
		fold_err = []
		# fold_size = []
		fstart_index = 0
		used_indices = []
		# print ('Input_vec: ', len(input_vec))
		for i in range(10):
			if i < extras:
				fold_size = int(len(input_vec)/10)+1
				
			else:
				fold_size  = int(len(input_vec)/10)

			temp = []
			temp_labels = []
			for f in range(fold_size):
				while True:
					val = np.random.randint(0, len(input_vec))
					if val not in used_indices:
						temp.append(input_vec[val])
						temp_labels.append(labels[val])
						used_indices.append(val)
						break
			fold.append(temp)
			fold_labels.append(temp_labels)


		for j in range(10):
			# test_data =[]
			train_data =[]
			train_labels = []
			test_data = fold[j]
			test_labels = fold_labels[j]
			
			# print ('test_data: ', len(test_data))
			for k in range(10):
				if k != j:
					train_data.extend(fold[k])
					train_labels.extend(fold_labels[k])	

			# print ('train_data: ', len(train_data))
			for l in range(epochs):
				erm_sum = 0
				for trainip, trainlb in zip(train_data, train_labels):
					predict_train = predict(trainip, weight)
					weight[1:] += learn_rate * abs((trainlb - predict_train)) * trainip
					weight[0] += learn_rate * abs((trainlb - predict_train))
					
			f_errsum = 0		
			for testip, testlb in zip(test_data, test_labels):
				predict_test = predict(testip, weight)
				f_errsum += abs(testlb - predict_test)

			f_err = f_errsum/len(train_data)
			fold_err.append(f_err)

			print("Fold", j, "Loss:", f_err)
		print("Loss:", sum(fold_err)/10)
		print("Accuracy:", 1 - sum(fold_err)/10)


def predict(vinput, w):
	sum_value = np.dot(vinput, w[1:]) + w[0]
	ans = call_activation(sum_value)
	return ans


def call_activation(sum_val):
	if sum_val >0:
		pr_label = 1
	else:
		pr_label =0
	return pr_label

def main():
	epochs = 100
	learning_rate = 0.1
	parse = argparse.ArgumentParser(description='Perceptron commandline')

	# Adding arguments
	parse.add_argument('--dataset', type = str, help='dataset file')
	parse.add_argument('--mode', type = str, help='ERM or ten fold')

	# Execute the parse_args() method
	args = parse.parse_args()

	data = np.loadtxt(args.dataset, delimiter=',', skiprows=1)

	#initalize weights
	train_input = []
	labels = []
	for i in data:
		train_input.append(i[0:-1])
		labels.append(i[len(i)-1])

	train(train_input, labels, args.mode, epochs, learning_rate)	

if __name__== '__main__':
	main()
	