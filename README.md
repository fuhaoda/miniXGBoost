# TODO (Draft)

Replace existing header files with the following ones. 

- `util.h`: define two iterator classes here. `ForwardIterator` and `BackwardIterator`, they will be used to traverse a given range of the matrix defined below. 

```
class ForwardIterator {
  next();  // move to the next one
  value(); // return the value associated with the current one
  start    // data member, start position
  end.     // data member, end location, [start, end) 
}; 
```

- `spare_matrix.h`. Define `Entry` and `SparseMatrix` class here. 
```
struct Entry {
  size_t index; // Column or row index depending on whether the entry is used in a CSR or CSC matrix
  float value; 
  operator <;   // Overload comparison operator, sort based on value. 
}; 

class SparseMatrix {
public: 
    
  
  vector<size_t> rowPtr_; 
  vector<size_t> colPtr_; 
  vector<Entry> rowData_; 
  vector<Entry> colData_;
  
  ForwardIterator getRow(size_t rowID, bool sorted = false, bool ascend = true); 
  BackwardIterator getRow(size_t 
  
}; 
```

- `tree.h`. Define `TreeNode` and `Tree`
```
struct TreeNode {
  size_t parent_;   // Index of the parent in the container
  size_t leftChild_; 
  size_t rightChild_;
  float sumGrad_; 
  float sumHess_; 
  
  float weight(); // Compute the weight 
  float gain(); // Compute the gain 
};

class Tree {
  Tree(
}; 
```
