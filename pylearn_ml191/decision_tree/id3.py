from collections import Counter
import numpy as np 

class TreeNode(object):
    def __init__(self, data=[], target=[],
                 children=[], entropy=None, depth=0):
        # Sub-data at node 
        self.data = data 
        self.target = target 
        
        # list of child node
        self.children = children
        
        # entropy of node
        self.entropy = entropy
        
        # depth of node
        self.depth = depth
        
        # atributte to split at node
        self.split_attribute = None 
        
        # label of any points at node, if it is leaf node.
        self.label = None 
            
    def set_properties(self, split_attribute, order):
        self.split_attribute = split_attribute
        self.order = order
    
    def set_label(self, label):
        self.label = label
    
    def print_tree(self):
        if self.depth == 0:
            print('[ROOT]', end='\n')
            
        if self.label != None:
            print(' --> class: {}'.format(self.label), end='\n')
            
        if self.order != None:
            for i, value in enumerate(self.order):
                print(' ' * 5 * self.children[i].depth, end='')
                if self.children[i].label != None:
                    print('[{} = {}]'.format(self.split_attribute, value), end='')
                else:
                    print('[{} = {}]'.format(self.split_attribute, value), end='\n')
                self.children[i].print_tree()   
                   
class DecisionTreeClassifier:
    """ Decision Tree Classifier with ID3 algorithm """
    def __init__(self, max_depth=5, min_samples_split=3, min_gain=1e-7):
        super().__init__()
        self.root = None 
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain

    def fit(self, data, target, attributes):
        """ Building decision tree with training data
        
        Parameters:
        ----------
            data: features data, (num_samples, num_features)
            target: target, (num_samples, 1)
            attributes: list of attributes, length = num_features
        
        Returns:
        --------
            None 
        """
        self.N = data.shape[0]
        self.data = data 
        self.attributes = attributes
        self.target = target 
        self.root = TreeNode(data=self.data, target=self.target, 
                             entropy=self._entropy(self.target), depth=0)
        
        queue = [self.root]
        while queue:
            node = queue.pop()
            if node.depth < self.max_depth or node.entropy < self.min_gain:
                node.children = self._split(node)
                if node.children == []: 
                    # leaf node
                    self._set_label(node)
                queue += node.children
            else:
                self._set_label(node)
                
    def _entropy(self, target):
        """ Get entropy from target value list """
        def entropy(freq):
            # convention: 0 * log(0) = 0
            freq = freq[np.array(freq).nonzero()[0]] # remove zeros values
            prob = freq / float(freq.sum())
            return -np.sum(prob * np.log(prob))
        freq = np.array(list(dict(Counter(target)).values()))
        return entropy(freq)
            
    def _set_label(self, node):
        """ Get label for a node if it is a leaf, simply chose by major voting  """
        target = list(node.target)
        label = max(set(target), key = target.count)
        node.set_label(label) 
    
    def _split(self, node):
        """ Split one node to some children nodes with id3 algorithm """
        data = node.data
        target = node.target
        best_gain = 0
        best_splits = []
        best_attribute = None
        order = None
        
        for i, att in enumerate(self.attributes):
            values = np.unique(data[:, i]).tolist()
            
            if len(values) == 1: 
                continue 
                
            splits = []
            for val in values: 
                sub_data = data[data[:, i] == val]
                sub_target = target[data[:, i] == val]
                splits.append((sub_data, sub_target))
            
            # Don't split if a node has too small number of points
            if min(map(len, [x[1] for x in splits])) < self.min_samples_split: 
                continue
            
            # --> Information Gain
            cond_entropy= 0
            for sub_data, sub_target in splits:
                cond_entropy += sub_data.shape[0] * self._entropy(sub_target) / data.shape[0]
            gain = node.entropy - cond_entropy 

            # If gain < min gain ==> don't split
            if gain < self.min_gain:
                continue

            if gain > best_gain:
                best_gain = gain 
                best_splits = splits
                best_attribute = att
                order = values
        
        node.set_properties(best_attribute, order)
        child_nodes = [TreeNode(data=sub_data, target=sub_target, 
                                entropy=self._entropy(sub_target), 
                                depth=node.depth + 1) 
                       for sub_data, sub_target in best_splits]
        
        return child_nodes
    
    def show_tree(self):
        """ Print tree learned from data """
        self.root.print_tree()
        
    def get_label(self, x):
        """ Get label of sample x from tree """
        # Starting at root node 
        node = self.root
        # Go to leaf node 
        while node.children: 
            idx_att = self.attributes.to_list().index(node.split_attribute)
            node = node.children[node.order.index(x[idx_att])]
        # Get label
        return node.label
    
    def predict(self, X):
        """ Predict label for new data """
        N = X.shape[0]
        return [self.get_label(X[i, :]) for i in range(N)]