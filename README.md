# 📘 DSA Cheat Sheet

This cheat sheet covers various Data Structures and Algorithms (DSA) patterns, techniques, and their use cases.

---

## 🔎 **Binary Search**
- **Category**: Algorithm
- **Tags**: #binary_search, #search
- **Similar To**: Linear Search, Ternary Search
- **Core Idea**: Binary search works on sorted arrays. It repeatedly divides the search interval in half and compares the target value to the middle element of the array.
- **Time Complexity**: O(log n)
- **Space Complexity**: O(1)
- **Limitations**: Works only on sorted arrays.
- **Use Cases**: Efficient searching in large sorted datasets.
- **Example**:
    - **Title**: Binary Search in a Sorted Array
    - **Code**:
      ```cpp
      int binarySearch(int arr[], int target, int n) {
          int low = 0, high = n - 1;
          while (low <= high) {
              int mid = low + (high - low) / 2;
              if (arr[mid] == target) return mid;
              else if (arr[mid] < target) low = mid + 1;
              else high = mid - 1;
          }
          return -1; // Target not found
      }
      ```
    - **Why It Works**: The array is divided into two halves, and the search continues in the half that may contain the target.

---

## 🔎 **Array**
- **Category**: Data Structure
- **Tags**: #array, #data_structure
- **Similar To**: List
- **Core Idea**: An array is a collection of elements identified by index or key, with a fixed size and contiguous memory allocation.
- **Time Complexity**: 
    - Access: O(1)
    - Insert/Delete: O(n) (worst case)
- **Space Complexity**: O(n)
- **Limitations**: Fixed size, inefficient for dynamic data.
- **Use Cases**: Storing a collection of elements that need to be accessed randomly.
- **Example**:
    - **Title**: Array Basics
    - **Code**:
      ```cpp
      int arr[5] = {1, 2, 3, 4, 5};
      cout << arr[2]; // Output: 3
      ```
    - **Why It Works**: Arrays store elements in contiguous memory locations, enabling constant time access to any element via its index.

---

## 🔎 **Left Right Pointers**
- **Category**: Technique
- **Tags**: #two_pointer, #array
- **Similar To**: Sliding Window, Fast Slow Pointers
- **Core Idea**: Two pointers are used to traverse the array from both ends towards the middle.
- **Time Complexity**: O(n)
- **Space Complexity**: O(1)
- **Limitations**: Not applicable to unsorted or unordered datasets.
- **Use Cases**: Solving problems like finding pairs in a sorted array that sum to a specific target.
- **Example**:
    - **Title**: Finding a Pair with a Given Sum
    - **Code**:
      ```cpp
      bool findPair(int arr[], int n, int target) {
          int left = 0, right = n - 1;
          while (left < right) {
              int sum = arr[left] + arr[right];
              if (sum == target) return true;
              else if (sum < target) left++;
              else right--;
          }
          return false;
      }
      ```
    - **Why It Works**: This method works because the array is sorted, and we adjust the pointers based on the sum's relation to the target.

---

## 🔎 **Fast Slow Pointers**
- **Category**: Technique
- **Tags**: #two_pointer, #linked_list
- **Similar To**: Left Right Pointers
- **Core Idea**: Two pointers are used to traverse a data structure at different speeds, often used to detect cycles in linked lists.
- **Time Complexity**: O(n)
- **Space Complexity**: O(1)
- **Limitations**: Assumes the presence of a cycle or loop in the data structure.
- **Use Cases**: Detecting cycles in linked lists, finding the middle element of a list.
- **Example**:
    - **Title**: Cycle Detection in Linked List
    - **Code**:
      ```cpp
      bool hasCycle(ListNode* head) {
          ListNode* slow = head;
          ListNode* fast = head;
          while (fast != NULL && fast->next != NULL) {
              slow = slow->next;
              fast = fast->next->next;
              if (slow == fast) return true;
          }
          return false;
      }
      ```
    - **Why It Works**: The slow pointer moves one step at a time, while the fast pointer moves two steps at a time. If a cycle exists, the two pointers will eventually meet.

---

## 🔎 **Sliding Window Fixed**
- **Category**: Algorithm
- **Tags**: #sliding_window, #array
- **Similar To**: Left Right Pointers
- **Core Idea**: A window of fixed size slides over an array or list to solve problems related to subarrays or substrings.
- **Time Complexity**: O(n)
- **Space Complexity**: O(1)
- **Limitations**: Only works with fixed-size subarrays or substrings.
- **Use Cases**: Finding the maximum or minimum sum of a subarray of fixed length.
- **Example**:
    - **Title**: Maximum Sum of Subarray of Size k
    - **Code**:
      ```cpp
      int maxSum(int arr[], int n, int k) {
          int max_sum = 0, window_sum = 0;
          for (int i = 0; i < k; i++) window_sum += arr[i];
          max_sum = window_sum;
          for (int i = k; i < n; i++) {
              window_sum += arr[i] - arr[i - k];
              max_sum = max(max_sum, window_sum);
          }
          return max_sum;
      }
      ```
    - **Why It Works**: The window sum is updated by adding the next element and removing the previous element, keeping the window size constant.

---

## 🔎 **Sliding Window Variable**
- **Category**: Algorithm
- **Tags**: #sliding_window, #array
- **Similar To**: Sliding Window Fixed
- **Core Idea**: A sliding window with a variable size is used to optimize solutions for problems that require flexibility in subarray or substring lengths.
- **Time Complexity**: O(n)
- **Space Complexity**: O(1)
- **Limitations**: Requires careful management of window size and conditions.
- **Use Cases**: Solving problems like finding the longest substring without repeating characters.
- **Example**:
    - **Title**: Longest Substring Without Repeating Characters
    - **Code**:
      ```cpp
      int lengthOfLongestSubstring(string s) {
          unordered_map<char, int> map;
          int left = 0, maxLength = 0;
          for (int right = 0; right < s.size(); right++) {
              if (map.find(s[right]) != map.end()) left = max(left, map[s[right]] + 1);
              map[s[right]] = right;
              maxLength = max(maxLength, right - left + 1);
          }
          return maxLength;
      }
      ```
    - **Why It Works**: The window adjusts dynamically by shifting the left pointer when a duplicate character is found, ensuring the substring remains valid.

---

## 🔎 **Boyer Moore**
- **Category**: Algorithm
- **Tags**: #string_search, #algorithm
- **Similar To**: Knuth-Morris-Pratt (KMP), Rabin-Karp
- **Core Idea**: A string matching algorithm that improves search performance by skipping over portions of the text based on pattern mismatches.
- **Time Complexity**: O(n)
- **Space Complexity**: O(m), where m is the length of the pattern.
- **Limitations**: More complex than other string matching algorithms.
- **Use Cases**: Searching for a pattern in a large text string.
- **Example**:
    - **Title**: Boyer-Moore String Search
    - **Code**:
      ```cpp
      void badCharacterHeuristic(string pattern, vector<int>& badChar) {
          int m = pattern.length();
          for (int i = 0; i < m; i++) badChar[pattern[i]] = i;
      }

      int boyerMooreSearch(string text, string pattern) {
          int n = text.length(), m = pattern.length();
          vector<int> badChar(256, -1);
          badCharacterHeuristic(pattern, badChar);
          
          int s = 0;
          while (s <= n - m) {
              int j = m - 1;
              while (j >= 0 && pattern[j] == text[s + j]) j--;
              if (j < 0) {
                  cout << "Pattern found at index " << s << endl;
                  s += (s + m < n) ? m - badChar[text[s + m]] : 1;
              } else {
                  s += max(1, j - badChar[text[s + j]]);
              }
          }
          return -1;
      }
      ```
    - **Why It Works**: The Boyer-Moore algorithm uses a mismatch heuristic to skip over sections of the text, making it efficient for large-scale text searches.

---

## 🔎 **Prefix Sum**
- **Category**: Technique
- **Tags**: #array, #prefix_sum
- **Similar To**: Cumulative Sum
- **Core Idea**: Prefix sum is a technique used to preprocess data, allowing quick range sum queries.
- **Time Complexity**: O(n) for preprocessing
- **Space Complexity**: O(n)
- **Limitations**: Only applicable for problems involving sums or cumulative calculations.
- **Use Cases**: Quickly answering range sum queries after an initial preprocessing step.
- **Example**:
    - **Title**: Prefix Sum for Range Queries
    - **Code**:
      ```cpp
      void buildPrefixSum(int arr[], int n, int prefixSum[]) {
          prefixSum[0] = arr[0];
          for (int i = 1; i < n; i++) {
              prefixSum[i] = prefixSum[i - 1] + arr[i];
          }
      }

      int getRangeSum(int prefixSum[], int left, int right) {
          if (left == 0) return prefixSum[right];
          return prefixSum[right] - prefixSum[left - 1];
      }
      ```
    - **Why It Works**: The prefix sum array stores cumulative sums up to each index, allowing for constant-time range sum queries.

---

## 🔎 **Difference Array**
- **Category**: Data Structure
- **Tags**: #array, #prefix_sum, #range_update
- **Similar To**: Prefix Sum
- **Core Idea**: A difference array allows efficient range updates in O(1) time by applying differences at specific indices and taking prefix sum to get the final array.
- **Time Complexity**: O(1) for update, O(n) for final computation
- **Space Complexity**: O(n)
- **Limitations**: Only applicable for additive updates. Needs final prefix sum to get actual array.
- **Use Cases**: Range increment updates, frequency tables, competitive programming.
- **Example**:
    - **Title**: Range Increment Using Difference Array
    - **Code**:
      ```cpp
      void rangeUpdate(int diff[], int l, int r, int val) {
          diff[l] += val;
          diff[r + 1] -= val;
      }

      void finalizeArray(int diff[], int n) {
          for (int i = 1; i < n; ++i)
              diff[i] += diff[i - 1];
      }
      ```
    - **Why It Works**: It marks the increment at start index and decrement at end+1. When prefix sum is taken, the net effect of all updates is realized.

---

## 🔗 **Linked List**
- **Category**: Data Structure
- **Tags**: #linked_list, #pointer, #dynamic_memory
- **Similar To**: Array (sequential storage)
- **Core Idea**: A linear data structure where each element points to the next. It allows dynamic memory allocation and insertion/deletion in O(1) time at head/tail.
- **Time Complexity**: O(n) for search, O(1) for insert/delete at head
- **Space Complexity**: O(n)
- **Limitations**: Random access is not possible. Overhead of pointers.
- **Use Cases**: Implementing queues, stacks, graphs, memory-efficient list operations.
- **Example**:
    - **Title**: Insert Node at Head
    - **Code**:
      ```cpp
      struct Node {
          int data;
          Node* next;
      };

      void insertAtHead(Node*& head, int val) {
          Node* newNode = new Node{val, head};
          head = newNode;
      }
      ```
    - **Why It Works**: A new node is created and made to point to current head. Head pointer is updated to this new node.

---

## 🔍 **Backtracking**
- **Category**: Algorithm
- **Tags**: #recursion, #backtracking, #dfs
- **Similar To**: DFS, Brute Force
- **Core Idea**: Recursively build the solution and backtrack upon reaching invalid states. It explores all possible options and undoes the last step to try new paths.
- **Time Complexity**: Exponential in worst case
- **Space Complexity**: O(n) (depth of recursion tree)
- **Limitations**: High time complexity for large inputs
- **Use Cases**: Permutations, combinations, puzzles like Sudoku, N-Queens
- **Example**:
    - **Title**: N-Queens Problem
    - **Code**:
      ```cpp
      void solve(int row, int n, vector<string>& board, vector<vector<string>>& res) {
          if (row == n) {
              res.push_back(board);
              return;
          }
          for (int col = 0; col < n; ++col) {
              if (isValid(board, row, col)) {
                  board[row][col] = 'Q';
                  solve(row + 1, n, board, res);
                  board[row][col] = '.';
              }
          }
      }
      ```
    - **Why It Works**: It tries to place a queen in each column, and backtracks if placing a queen leads to an invalid board.

---

## 🔐 **Hash Map**
- **Category**: Data Structure
- **Tags**: #hashmap, #key_value, #unordered
- **Similar To**: TreeMap, Dictionary
- **Core Idea**: Stores key-value pairs with average O(1) access time using a hash function.
- **Time Complexity**: O(1) average, O(n) worst case (hash collision)
- **Space Complexity**: O(n)
- **Limitations**: Not ordered, collisions may degrade performance
- **Use Cases**: Frequency count, caching, fast lookups
- **Example**:
    - **Title**: Frequency Count
    - **Code**:
      ```cpp
      unordered_map<int, int> freq;
      for (int num : nums) {
          freq[num]++;
      }
      ```
    - **Why It Works**: `unordered_map` allows counting occurrences in constant time.

---

## 🧮 **Hash Set**
- **Category**: Data Structure
- **Tags**: #set, #hashing, #unique
- **Similar To**: Array, Hash Map
- **Core Idea**: Stores unique elements with average O(1) operations using hashing.
- **Time Complexity**: O(1) average for insert/find/delete
- **Space Complexity**: O(n)
- **Limitations**: No order, duplicates not allowed
- **Use Cases**: Removing duplicates, membership test, fast lookups
- **Example**:
    - **Title**: Remove Duplicates from Array
    - **Code**:
      ```cpp
      unordered_set<int> s(nums.begin(), nums.end());
      vector<int> unique(s.begin(), s.end());
      ```
    - **Why It Works**: `unordered_set` inserts only unique elements.

---

## 🔢 **Hash Counting**
- **Category**: Data Structure
- **Tags**: #hashmap, #frequency, #counting
- **Similar To**: Frequency Map, Dictionary
- **Core Idea**: Use a hash map (unordered_map in C++) to count the number of occurrences of elements.
- **Time Complexity**: O(n)
- **Space Complexity**: O(n)
- **Limitations**: High memory usage for large ranges of keys.
- **Use Cases**: Counting word frequency, finding duplicates, majority element problems.
- **Example**:
    - **Title**: Count Frequencies of Elements in an Array
    - **Code**:
      ```cpp
      unordered_map<int, int> countFrequencies(vector<int>& nums) {
          unordered_map<int, int> freq;
          for (int num : nums) {
              freq[num]++;
          }
          return freq;
      }
      ```
    - **Why It Works**: Each element is counted using a hash table with constant time access.

---

## 🧵 **String**
- **Category**: Data Structure
- **Tags**: #string, #manipulation
- **Similar To**: Character Array
- **Core Idea**: Strings are sequences of characters that support manipulation like slicing, comparing, and building new strings.
- **Time Complexity**: O(n) for most operations
- **Space Complexity**: O(n)
- **Limitations**: Immutable in some languages; operations like concatenation can be costly if not optimized.
- **Use Cases**: Pattern matching, parsing, tokenizing.
- **Example**:
    - **Title**: Reverse a String
    - **Code**:
      ```cpp
      string reverseString(string s) {
          int l = 0, r = s.size() - 1;
          while (l < r) swap(s[l++], s[r--]);
          return s;
      }
      ```
    - **Why It Works**: Two-pointer approach efficiently reverses the string in-place.

---

## 🔍 **KMP (Knuth-Morris-Pratt)**
- **Category**: Algorithm
- **Tags**: #string_matching, #prefix_function
- **Similar To**: Rabin-Karp, Boyer-Moore
- **Core Idea**: Use a prefix table (LPS array) to avoid redundant comparisons during string matching.
- **Time Complexity**: O(n + m)
- **Space Complexity**: O(m)
- **Limitations**: Slightly complex to implement.
- **Use Cases**: Pattern searching in texts, DNA sequence analysis.
- **Example**:
    - **Title**: KMP String Matching
    - **Code**:
      ```cpp
      vector<int> computeLPS(string pattern) {
          int m = pattern.size(), len = 0;
          vector<int> lps(m, 0);
          for (int i = 1; i < m; ) {
              if (pattern[i] == pattern[len]) lps[i++] = ++len;
              else if (len) len = lps[len - 1];
              else lps[i++] = 0;
          }
          return lps;
      }

      vector<int> KMPSearch(string text, string pattern) {
          vector<int> lps = computeLPS(pattern), result;
          int i = 0, j = 0;
          while (i < text.size()) {
              if (pattern[j] == text[i]) { i++; j++; }
              if (j == pattern.size()) {
                  result.push_back(i - j);
                  j = lps[j - 1];
              } else if (i < text.size() && pattern[j] != text[i]) {
                  if (j) j = lps[j - 1];
                  else i++;
              }
          }
          return result;
      }
      ```
    - **Why It Works**: The LPS array allows skipping characters to reduce time complexity.

---

## 🌲 **Tree Traversal**
- **Category**: Algorithm
- **Tags**: #tree, #dfs, #bfs
- **Similar To**: Graph Traversal
- **Core Idea**: Traverse a tree using preorder, inorder, postorder (DFS) or level order (BFS).
- **Time Complexity**: O(n)
- **Space Complexity**: O(h) for DFS, O(n) for BFS
- **Limitations**: Recursive depth can hit stack limit in large/deep trees.
- **Use Cases**: Tree evaluation, serialization, expression trees.
- **Example**:
    - **Title**: Inorder Traversal (Recursive)
    - **Code**:
      ```cpp
      void inorder(TreeNode* root, vector<int>& res) {
          if (!root) return;
          inorder(root->left, res);
          res.push_back(root->val);
          inorder(root->right, res);
      }
      ```
    - **Why It Works**: Recursively visits left subtree, root, and right subtree in order.

---

## 🌳 **Tree Feature**
- **Category**: Pattern
- **Tags**: #tree, #properties
- **Similar To**: Tree Traversal
- **Core Idea**: Exploit tree properties like height, diameter, symmetry, balance.
- **Time Complexity**: O(n)
- **Space Complexity**: O(h)
- **Limitations**: Must handle null nodes and balance conditions.
- **Use Cases**: Checking tree validity, balancing trees.
- **Example**:
    - **Title**: Height of Binary Tree
    - **Code**:
      ```cpp
      int height(TreeNode* root) {
          if (!root) return 0;
          return 1 + max(height(root->left), height(root->right));
      }
      ```
    - **Why It Works**: Recursively finds the height of left and right subtrees.

---

## 🌐 **Tree BFS**
- **Category**: Algorithm
- **Tags**: #tree, #bfs, #levelorder
- **Similar To**: Graph BFS
- **Core Idea**: Use a queue to traverse nodes level by level.
- **Time Complexity**: O(n)
- **Space Complexity**: O(n)
- **Limitations**: Can be memory-intensive for wide trees.
- **Use Cases**: Finding minimum depth, level order traversal.
- **Example**:
    - **Title**: Level Order Traversal
    - **Code**:
      ```cpp
      vector<vector<int>> levelOrder(TreeNode* root) {
          vector<vector<int>> res;
          if (!root) return res;
          queue<TreeNode*> q;
          q.push(root);
          while (!q.empty()) {
              int size = q.size();
              vector<int> level;
              for (int i = 0; i < size; ++i) {
                  TreeNode* node = q.front(); q.pop();
                  level.push_back(node->val);
                  if (node->left) q.push(node->left);
                  if (node->right) q.push(node->right);
              }
              res.push_back(level);
          }
          return res;
      }
      ```
    - **Why It Works**: Processes nodes level by level using a queue.

---

## 🌳 **Tree Modification**
- **Category**: Pattern
- **Tags**: #tree, #recursion, #dfs
- **Similar To**: DFS Traversal, Recursion
- **Core Idea**: Modify tree structure or node values using recursive or iterative DFS/Preorder/Postorder techniques.
- **Time Complexity**: O(n)
- **Space Complexity**: O(h) for recursion stack (h = height of tree)
- **Limitations**: Stack overflow for deep trees if not using tail recursion.
- **Use Cases**: Inverting a tree, trimming BST, modifying values based on subtree.
- **Example**:
    - **Title**: Invert Binary Tree
    - **Code**:
      ```cpp
      TreeNode* invertTree(TreeNode* root) {
          if (!root) return nullptr;
          swap(root->left, root->right);
          invertTree(root->left);
          invertTree(root->right);
          return root;
      }
      ```
    - **Why It Works**: Swaps left and right children recursively to invert the tree structure.

---

## 🌲 **BST (Binary Search Tree)**
- **Category**: Data Structure
- **Tags**: #bst, #tree
- **Similar To**: Binary Tree, AVL Tree
- **Core Idea**: Tree where left child < parent < right child, allows efficient searching and insertion.
- **Time Complexity**: O(log n) average, O(n) worst-case
- **Space Complexity**: O(h) for recursion stack
- **Limitations**: Unbalanced trees degrade to linked lists (O(n) ops).
- **Use Cases**: Range queries, dynamic datasets with ordering.
- **Example**:
    - **Title**: Insert into BST
    - **Code**:
      ```cpp
      TreeNode* insertBST(TreeNode* root, int val) {
          if (!root) return new TreeNode(val);
          if (val < root->val) root->left = insertBST(root->left, val);
          else root->right = insertBST(root->right, val);
          return root;
      }
      ```
    - **Why It Works**: Inserts element preserving the BST properties recursively.

---

## 🌐 **Trie**
- **Category**: Data Structure
- **Tags**: #string, #prefix_tree, #trie
- **Similar To**: HashMap, Tree
- **Core Idea**: Tree-like structure for storing strings with common prefixes.
- **Time Complexity**: O(L) for insert/search (L = length of word)
- **Space Complexity**: O(N * L * 26)
- **Limitations**: High memory usage
- **Use Cases**: Autocomplete, Spell checker, Prefix search
- **Example**:
    - **Title**: Insert and Search in Trie
    - **Code**:
      ```cpp
      struct TrieNode {
          TrieNode* children[26] = {};
          bool isEnd = false;
      };

      class Trie {
          TrieNode* root;
      public:
          Trie() { root = new TrieNode(); }
          void insert(string word) {
              TrieNode* node = root;
              for (char c : word) {
                  if (!node->children[c - 'a'])
                      node->children[c - 'a'] = new TrieNode();
                  node = node->children[c - 'a'];
              }
              node->isEnd = true;
          }

          bool search(string word) {
              TrieNode* node = root;
              for (char c : word) {
                  if (!node->children[c - 'a']) return false;
                  node = node->children[c - 'a'];
              }
              return node->isEnd;
          }
      };
      ```
    - **Why It Works**: Follows each character in the Trie, creating nodes as needed.

---

## 📚 **Stack**
- **Category**: Data Structure
- **Tags**: #stack, #lifo
- **Similar To**: Queue, Array
- **Core Idea**: LIFO (Last-In-First-Out) structure, elements are added and removed from the top.
- **Time Complexity**: O(1) for push/pop
- **Space Complexity**: O(n)
- **Limitations**: Limited access to only top element
- **Use Cases**: Reversal, recursion simulation, expression evaluation
- **Example**:
    - **Title**: Valid Parentheses
    - **Code**:
      ```cpp
      bool isValid(string s) {
          stack<char> st;
          for (char c : s) {
              if (c == '(' || c == '{' || c == '[') st.push(c);
              else {
                  if (st.empty()) return false;
                  char t = st.top(); st.pop();
                  if ((c == ')' && t != '(') ||
                      (c == '}' && t != '{') ||
                      (c == ']' && t != '['))
                      return false;
              }
          }
          return st.empty();
      }
      ```
    - **Why It Works**: Tracks opening brackets and ensures they match the closing ones in correct order.

---

## 🧱 **Stack Monotonic**
- **Category**: Pattern
- **Tags**: #stack, #monotonic
- **Similar To**: Sliding Window, Two Pointers
- **Core Idea**: Use stack to maintain increasing or decreasing sequence for next greater/smaller problems.
- **Time Complexity**: O(n)
- **Space Complexity**: O(n)
- **Limitations**: Only works for problems involving ordering relations
- **Use Cases**: Next Greater Element, Histogram Area
- **Example**:
    - **Title**: Next Greater Element
    - **Code**:
      ```cpp
      vector<int> nextGreater(vector<int>& nums) {
          stack<int> st;
          vector<int> res(nums.size(), -1);
          for (int i = 0; i < nums.size(); ++i) {
              while (!st.empty() && nums[i] > nums[st.top()]) {
                  res[st.top()] = nums[i];
                  st.pop();
              }
              st.push(i);
          }
          return res;
      }
      ```
    - **Why It Works**: Keeps track of indices of elements for which a greater element is yet to be found.

---

## 📬 **Queue**
- **Category**: Data Structure
- **Tags**: #queue, #fifo
- **Similar To**: Stack
- **Core Idea**: FIFO (First-In-First-Out) structure where elements are added at rear and removed from front.
- **Time Complexity**: O(1) for enqueue/dequeue (amortized with circular buffer)
- **Space Complexity**: O(n)
- **Limitations**: Inefficient index-based access
- **Use Cases**: Task scheduling, level order traversal
- **Example**:
    - **Title**: Level Order Traversal
    - **Code**:
      ```cpp
      vector<vector<int>> levelOrder(TreeNode* root) {
          vector<vector<int>> res;
          if (!root) return res;
          queue<TreeNode*> q;
          q.push(root);
          while (!q.empty()) {
              int n = q.size();
              vector<int> level;
              for (int i = 0; i < n; ++i) {
                  TreeNode* node = q.front(); q.pop();
                  level.push_back(node->val);
                  if (node->left) q.push(node->left);
                  if (node->right) q.push(node->right);
              }
              res.push_back(level);
          }
          return res;
      }
      ```
    - **Why It Works**: Processes all nodes level-by-level using queue.

---

## 📉 **Queue Monotonic**
- **Category**: Pattern
- **Tags**: #queue, #monotonic
- **Similar To**: Sliding Window
- **Core Idea**: Maintains elements in a window with ordered value using deque to find min/max efficiently.
- **Time Complexity**: O(n)
- **Space Complexity**: O(k) (for window size k)
- **Limitations**: Only useful for sliding window max/min
- **Use Cases**: Max in Sliding Window
- **Example**:
    - **Title**: Sliding Window Maximum
    - **Code**:
      ```cpp
      vector<int> maxSlidingWindow(vector<int>& nums, int k) {
          deque<int> dq;
          vector<int> res;
          for (int i = 0; i < nums.size(); ++i) {
              if (!dq.empty() && dq.front() == i - k) dq.pop_front();
              while (!dq.empty() && nums[i] >= nums[dq.back()]) dq.pop_back();
              dq.push_back(i);
              if (i >= k - 1) res.push_back(nums[dq.front()]);
          }
          return res;
      }
      ```
    - **Why It Works**: Maintains decreasing deque so front always has the largest element in the window.

---

## 🔄 **DP Basic**
- **Category**: Pattern
- **Tags**: #dp, #recursion, #memoization
- **Similar To**: Recursion with Memoization
- **Core Idea**: Use recursion to solve subproblems and cache their results to avoid redundant calculations.
- **Time Complexity**: O(n)
- **Space Complexity**: O(n) (for memoization)
- **Limitations**: High space usage for deep recursion stacks.
- **Use Cases**: Fibonacci, climbing stairs, minimum cost problems.
- **Example**:
  - **Title**: Fibonacci using DP (Top-Down)
  - **Code**:
    ```cpp
    int fib(int n, vector<int>& dp) {
        if (n <= 1) return n;
        if (dp[n] != -1) return dp[n];
        return dp[n] = fib(n - 1, dp) + fib(n - 2, dp);
    }
    ```
  - **Why It Works**: Saves results of subproblems so they're not recalculated.

---

## 🧮 **DP 2D**
- **Category**: Pattern
- **Tags**: #dp, #2d_dp
- **Similar To**: Matrix-based Recursion
- **Core Idea**: Use a 2D table to represent states of subproblems with two changing parameters.
- **Time Complexity**: O(n * m)
- **Space Complexity**: O(n * m)
- **Limitations**: Large memory usage.
- **Use Cases**: Grid problems, pathfinding in matrices, edit distance.
- **Example**:
  - **Title**: Unique Paths in Grid
  - **Code**:
    ```cpp
    int uniquePaths(int m, int n) {
        vector<vector<int>> dp(m, vector<int>(n, 1));
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }
    ```
  - **Why It Works**: Each cell depends on top and left cells, building from base case.

---

## 💹 **DP Stock**
- **Category**: Pattern
- **Tags**: #dp, #stock, #greedy
- **Similar To**: Greedy, Kadane
- **Core Idea**: Use states to track buy/sell actions over days.
- **Time Complexity**: O(n)
- **Space Complexity**: O(1) or O(n)
- **Limitations**: Complex state tracking.
- **Use Cases**: Max profit with/without transaction limits.
- **Example**:
  - **Title**: Best Time to Buy and Sell Stock
  - **Code**:
    ```cpp
    int maxProfit(vector<int>& prices) {
        int minPrice = INT_MAX, maxProfit = 0;
        for (int price : prices) {
            minPrice = min(minPrice, price);
            maxProfit = max(maxProfit, price - minPrice);
        }
        return maxProfit;
    }
    ```
  - **Why It Works**: Tracks lowest price and max profit on the fly.

---

## 🎒 **DP 01 Knapsack**
- **Category**: Pattern
- **Tags**: #dp, #knapsack
- **Similar To**: Subset Sum
- **Core Idea**: Use DP to track max value for weight constraints.
- **Time Complexity**: O(n * W)
- **Space Complexity**: O(n * W)
- **Limitations**: Only whole items, not fractions.
- **Use Cases**: Budget problems, subset optimization.
- **Example**:
  - **Title**: 0/1 Knapsack
  - **Code**:
    ```cpp
    int knapsack(int W, vector<int>& wt, vector<int>& val, int n) {
        vector<vector<int>> dp(n+1, vector<int>(W+1, 0));
        for(int i=1;i<=n;i++) {
            for(int w=0;w<=W;w++) {
                if(wt[i-1]<=w)
                    dp[i][w] = max(val[i-1] + dp[i-1][w-wt[i-1]], dp[i-1][w]);
                else
                    dp[i][w] = dp[i-1][w];
            }
        }
        return dp[n][W];
    }
    ```
  - **Why It Works**: Considers both taking and not taking the item.

---

## 🎒 **DP Unbounded Knapsack**
- **Category**: Pattern
- **Tags**: #dp, #knapsack
- **Similar To**: 01 Knapsack
- **Core Idea**: Reuse items multiple times, unlike 01 Knapsack.
- **Time Complexity**: O(n * W)
- **Space Complexity**: O(W)
- **Limitations**: Infinite item count assumed.
- **Use Cases**: Coin change, cutting rod.
- **Example**:
  - **Title**: Coin Change (Min Coins)
  - **Code**:
    ```cpp
    int coinChange(vector<int>& coins, int amount) {
        vector<int> dp(amount + 1, 1e9);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int coin : coins) {
                if (i - coin >= 0)
                    dp[i] = min(dp[i], 1 + dp[i - coin]);
            }
        }
        return dp[amount] == 1e9 ? -1 : dp[amount];
    }
    ```
  - **Why It Works**: Builds answer by reusing items repeatedly.

---

## 📈 **DP Kadane**
- **Category**: Algorithm
- **Tags**: #dp, #kadane, #subarray
- **Similar To**: Sliding Window
- **Core Idea**: Track current subarray sum and reset on negative.
- **Time Complexity**: O(n)
- **Space Complexity**: O(1)
- **Limitations**: Only works for contiguous subarrays.
- **Use Cases**: Max subarray sum.
- **Example**:
  - **Title**: Maximum Subarray
  - **Code**:
    ```cpp
    int maxSubArray(vector<int>& nums) {
        int maxSum = nums[0], curSum = nums[0];
        for (int i = 1; i < nums.size(); i++) {
            curSum = max(nums[i], curSum + nums[i]);
            maxSum = max(maxSum, curSum);
        }
        return maxSum;
    }
    ```
  - **Why It Works**: Keeps best running sum; resets if it drops below 0.

---

## 🧠 **DP Interval**
- **Category**: Dynamic Programming
- **Tags**: #dp, #interval_dp
- **Similar To**: Matrix Chain Multiplication, Palindrome Partitioning
- **Core Idea**: Use DP to find the optimal solution over all intervals by dividing problems into subintervals and building up.
- **Time Complexity**: O(n^3)
- **Space Complexity**: O(n^2)
- **Limitations**: High time complexity for large intervals.
- **Use Cases**: Matrix Chain Multiplication, Burst Balloons, Merging Stones.
- **Example**:
    - **Title**: Burst Balloons
    - **Code**:
      ```cpp
      int maxCoins(vector<int>& nums) {
          int n = nums.size();
          vector<vector<int>> dp(n + 2, vector<int>(n + 2, 0));
          nums.insert(nums.begin(), 1);
          nums.push_back(1);
          for (int len = 1; len <= n; ++len) {
              for (int left = 1; left <= n - len + 1; ++left) {
                  int right = left + len - 1;
                  for (int k = left; k <= right; ++k) {
                      dp[left][right] = max(dp[left][right],
                          nums[left - 1] * nums[k] * nums[right + 1] +
                          dp[left][k - 1] + dp[k + 1][right]);
                  }
              }
          }
          return dp[1][n];
      }
      ```
    - **Why It Works**: We recursively choose the best balloon to burst last in every interval and store results to avoid recomputation.

---

## 🧠 **DP LCS**
- **Category**: Dynamic Programming
- **Tags**: #dp, #lcs
- **Similar To**: Edit Distance, Longest Palindromic Subsequence
- **Core Idea**: Build a DP table where dp[i][j] stores the LCS of first i and j characters of two strings.
- **Time Complexity**: O(n * m)
- **Space Complexity**: O(n * m)
- **Limitations**: Not space-efficient for very long strings.
- **Use Cases**: File comparison, diff utilities.
- **Example**:
    - **Title**: Longest Common Subsequence
    - **Code**:
      ```cpp
      int longestCommonSubsequence(string text1, string text2) {
          int n = text1.size(), m = text2.size();
          vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));
          for (int i = 1; i <= n; ++i) {
              for (int j = 1; j <= m; ++j) {
                  if (text1[i - 1] == text2[j - 1])
                      dp[i][j] = 1 + dp[i - 1][j - 1];
                  else
                      dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
              }
          }
          return dp[n][m];
      }
      ```
    - **Why It Works**: Subproblems build upon previous LCS values, combining them to find the optimal length.

---

## 🧠 **DP LIS**
- **Category**: Dynamic Programming
- **Tags**: #dp, #lis
- **Similar To**: Longest Increasing Subsequence
- **Core Idea**: Maintain an array where each entry stores the length of the LIS ending at that index.
- **Time Complexity**: O(n log n)
- **Space Complexity**: O(n)
- **Limitations**: Doesn’t give actual sequence in O(n log n) version.
- **Use Cases**: Sorting sequences, patience sorting.
- **Example**:
    - **Title**: Longest Increasing Subsequence
    - **Code**:
      ```cpp
      int lengthOfLIS(vector<int>& nums) {
          vector<int> lis;
          for (int num : nums) {
              auto it = lower_bound(lis.begin(), lis.end(), num);
              if (it == lis.end()) lis.push_back(num);
              else *it = num;
          }
          return lis.size();
      }
      ```
    - **Why It Works**: Uses binary search to place elements efficiently while maintaining a valid increasing sequence.

---

## ⚡ **Greedy**
- **Category**: Pattern
- **Tags**: #greedy
- **Similar To**: DP (for optimization), Intervals
- **Core Idea**: Make the locally optimal choice at each step hoping it leads to the global optimum.
- **Time Complexity**: Usually O(n log n) or O(n)
- **Space Complexity**: O(1) or O(n)
- **Limitations**: Doesn't always yield the correct solution.
- **Use Cases**: Activity selection, Huffman coding, coin change (specific denominations).
- **Example**:
    - **Title**: Activity Selection
    - **Code**:
      ```cpp
      bool cmp(pair<int, int>& a, pair<int, int>& b) {
          return a.second < b.second;
      }
      int activitySelection(vector<pair<int, int>>& activities) {
          sort(activities.begin(), activities.end(), cmp);
          int count = 1, end = activities[0].second;
          for (int i = 1; i < activities.size(); ++i) {
              if (activities[i].first >= end) {
                  count++;
                  end = activities[i].second;
              }
          }
          return count;
      }
      ```
    - **Why It Works**: Greedily chooses activities with earliest finishing time to maximize the number of non-overlapping activities.

---

## 🎨 **Intervals**
- **Category**: Pattern
- **Tags**: #intervals, #greedy, #sorting
- **Similar To**: Sweep Line Algorithm, Merge Intervals
- **Core Idea**: Involves sorting intervals based on start or end points, then merging, inserting, or selecting them based on overlaps.
- **Time Complexity**: O(n log n) for sorting
- **Space Complexity**: O(n) (can be O(1) depending on in-place implementation)
- **Limitations**: Edge cases with overlapping intervals; requires careful sorting.
- **Use Cases**: Merging overlapping intervals, meeting room scheduling, range coverage.
- **Example**:
    - **Title**: Merge Overlapping Intervals
    - **Code**:
      ```cpp
      vector<vector<int>> merge(vector<vector<int>>& intervals) {
          sort(intervals.begin(), intervals.end());
          vector<vector<int>> merged;
          for (auto& interval : intervals) {
              if (merged.empty() || merged.back()[1] < interval[0])
                  merged.push_back(interval);
              else
                  merged.back()[1] = max(merged.back()[1], interval[1]);
          }
          return merged;
      }
      ```
    - **Why It Works**: Sorting helps detect overlaps; merged intervals are extended only when necessary.

---

## 🌐 **Graph Flood Fill**
- **Category**: Algorithm
- **Tags**: #graph, #dfs, #bfs
- **Similar To**: DFS, BFS, Connected Components
- **Core Idea**: Explore connected nodes in a grid/graph and change their state (e.g., color).
- **Time Complexity**: O(n × m)
- **Space Complexity**: O(n × m) (recursion stack or queue)
- **Limitations**: Stack overflow in DFS for large grids.
- **Use Cases**: Image fill, islands counting, region detection.
- **Example**:
    - **Title**: DFS Flood Fill
    - **Code**:
      ```cpp
      void dfs(vector<vector<int>>& image, int r, int c, int color, int newColor) {
          if (r < 0 || r >= image.size() || c < 0 || c >= image[0].size() || image[r][c] != color)
              return;
          image[r][c] = newColor;
          dfs(image, r+1, c, color, newColor);
          dfs(image, r-1, c, color, newColor);
          dfs(image, r, c+1, color, newColor);
          dfs(image, r, c-1, color, newColor);
      }

      vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int newColor) {
          if (image[sr][sc] != newColor)
              dfs(image, sr, sc, image[sr][sc], newColor);
          return image;
      }
      ```
    - **Why It Works**: Recursively changes each connected pixel to the new color.

---

## 🔍 **Graph BFS**
- **Category**: Algorithm
- **Tags**: #graph, #bfs
- **Similar To**: DFS, Shortest Path (unweighted)
- **Core Idea**: Uses a queue to explore nodes level by level.
- **Time Complexity**: O(V + E)
- **Space Complexity**: O(V)
- **Limitations**: High memory usage in dense graphs.
- **Use Cases**: Shortest path in unweighted graphs, level-order traversal.
- **Example**:
    - **Title**: BFS on an Undirected Graph
    - **Code**:
      ```cpp
      void bfs(int start, vector<vector<int>>& adj, vector<bool>& visited) {
          queue<int> q;
          q.push(start);
          visited[start] = true;

          while (!q.empty()) {
              int node = q.front(); q.pop();
              for (int neighbor : adj[node]) {
                  if (!visited[neighbor]) {
                      visited[neighbor] = true;
                      q.push(neighbor);
                  }
              }
          }
      }
      ```
    - **Why It Works**: Explores all nodes at the current depth before moving deeper.

---

## 📚 **Graph Topological Sort**
- **Category**: Algorithm
- **Tags**: #graph, #toposort, #dag
- **Similar To**: Kahn's Algorithm, DFS
- **Core Idea**: Orders nodes of a DAG such that for every directed edge u → v, u comes before v.
- **Time Complexity**: O(V + E)
- **Space Complexity**: O(V + E)
- **Limitations**: Only works for DAGs.
- **Use Cases**: Task scheduling, course prerequisites.
- **Example**:
    - **Title**: Topological Sort using Kahn's Algorithm
    - **Code**:
      ```cpp
      vector<int> topologicalSort(int V, vector<vector<int>>& adj) {
          vector<int> inDegree(V, 0);
          for (int u = 0; u < V; u++)
              for (int v : adj[u])
                  inDegree[v]++;

          queue<int> q;
          for (int i = 0; i < V; i++)
              if (inDegree[i] == 0) q.push(i);

          vector<int> topo;
          while (!q.empty()) {
              int node = q.front(); q.pop();
              topo.push_back(node);
              for (int neighbor : adj[node]) {
                  if (--inDegree[neighbor] == 0)
                      q.push(neighbor);
              }
          }
          return topo;
      }
      ```
    - **Why It Works**: Maintains a queue of nodes with zero in-degree to build the order.

---

## 🤝 **Graph Union Find**
- **Category**: Data Structure
- **Tags**: #graph, #disjoint_set
- **Similar To**: Connected Components, Kruskal’s Algorithm
- **Core Idea**: Maintains disjoint sets to detect cycles or check connectivity.
- **Time Complexity**: O(α(n)) per operation (with path compression and union by rank)
- **Space Complexity**: O(n)
- **Limitations**: Inefficient without path compression.
- **Use Cases**: Cycle detection, Kruskal’s MST, dynamic connectivity.
- **Example**:
    - **Title**: Union-Find with Path Compression
    - **Code**:
      ```cpp
      vector<int> parent, rank;

      int find(int x) {
          if (parent[x] != x)
              parent[x] = find(parent[x]);
          return parent[x];
      }

      void unionSet(int x, int y) {
          int rootX = find(x), rootY = find(y);
          if (rootX == rootY) return;
          if (rank[rootX] < rank[rootY]) parent[rootX] = rootY;
          else if (rank[rootX] > rank[rootY]) parent[rootY] = rootX;
          else {
              parent[rootY] = rootX;
              rank[rootX]++;
          }
      }

      void init(int n) {
          parent.resize(n);
          rank.resize(n, 0);
          for (int i = 0; i < n; ++i) parent[i] = i;
      }
      ```
    - **Why It Works**: Uses path compression and union by rank to optimize operations.

---

## 🔀 **Graph Shortest Path**
- **Category**: Algorithm
- **Tags**: #graph, #shortest_path, #dijkstra, #bfs
- **Similar To**: Dijkstra, Bellman-Ford, A*
- **Core Idea**: Finds the shortest path from a source node to all other nodes in a graph. Can be solved using Dijkstra's (greedy) or BFS for unweighted graphs.
- **Time Complexity**: O((V + E) log V) with Dijkstra using a priority queue
- **Space Complexity**: O(V + E)
- **Limitations**: Dijkstra doesn't work with negative weights.
- **Use Cases**: Navigation systems, network routing, game pathfinding
- **Example**:
    - **Title**: Dijkstra's Algorithm
    - **Code**:
      ```cpp
      vector<int> dijkstra(int V, vector<pair<int,int>> adj[]) {
          priority_queue<pair<int, int>, vector<pair<int,int>>, greater<pair<int,int>>> pq;
          vector<int> dist(V, INT_MAX);
          dist[0] = 0;
          pq.push({0, 0});

          while (!pq.empty()) {
              int u = pq.top().second;
              int d = pq.top().first;
              pq.pop();

              for (auto &[v, wt] : adj[u]) {
                  if (dist[v] > d + wt) {
                      dist[v] = d + wt;
                      pq.push({dist[v], v});
                  }
              }
          }
          return dist;
      }
      ```
    - **Why It Works**: The algorithm always expands the shortest known path first, using a priority queue.

---

## 📉 **Graph Bellman Ford**
- **Category**: Algorithm
- **Tags**: #graph, #shortest_path, #bellmanford
- **Similar To**: Dijkstra
- **Core Idea**: Finds the shortest paths from the source to all vertices, handles negative weights.
- **Time Complexity**: O(V * E)
- **Space Complexity**: O(V)
- **Limitations**: Slower than Dijkstra; doesn't work with negative weight cycles.
- **Use Cases**: Currency exchange, time-travel-like dependencies
- **Example**:
    - **Title**: Bellman-Ford Algorithm
    - **Code**:
      ```cpp
      bool bellmanFord(int V, vector<vector<int>> &edges, int src, vector<int> &dist) {
          dist.assign(V, INT_MAX);
          dist[src] = 0;

          for (int i = 1; i < V; ++i) {
              for (auto &edge : edges) {
                  int u = edge[0], v = edge[1], wt = edge[2];
                  if (dist[u] != INT_MAX && dist[u] + wt < dist[v]) {
                      dist[v] = dist[u] + wt;
                  }
              }
          }

          // Check for negative-weight cycles
          for (auto &edge : edges) {
              int u = edge[0], v = edge[1], wt = edge[2];
              if (dist[u] != INT_MAX && dist[u] + wt < dist[v]) return false;
          }
          return true;
      }
      ```
    - **Why It Works**: It relaxes all edges V-1 times to ensure shortest distances and detects cycles in a final pass.

---

## 🌲 **Graph Minimum Spanning Tree**
- **Category**: Algorithm
- **Tags**: #graph, #mst, #kruskal, #prim
- **Similar To**: Shortest Path
- **Core Idea**: Connect all nodes with minimum total edge weight, no cycles.
- **Time Complexity**: O(E log V) using Kruskal + DSU
- **Space Complexity**: O(V + E)
- **Limitations**: Only works on undirected connected graphs
- **Use Cases**: Network design, clustering
- **Example**:
    - **Title**: Kruskal's Algorithm
    - **Code**:
      ```cpp
      struct DSU {
          vector<int> parent, rank;
          DSU(int n) : parent(n), rank(n, 0) {
              for (int i = 0; i < n; ++i) parent[i] = i;
          }
          int find(int x) {
              return parent[x] == x ? x : parent[x] = find(parent[x]);
          }
          bool unite(int x, int y) {
              int rx = find(x), ry = find(y);
              if (rx == ry) return false;
              if (rank[rx] < rank[ry]) swap(rx, ry);
              parent[ry] = rx;
              if (rank[rx] == rank[ry]) ++rank[rx];
              return true;
          }
      };

      int kruskal(int V, vector<vector<int>> &edges) {
          sort(edges.begin(), edges.end(), [](auto &a, auto &b) {
              return a[2] < b[2];
          });
          DSU dsu(V);
          int cost = 0;
          for (auto &e : edges) {
              if (dsu.unite(e[0], e[1])) cost += e[2];
          }
          return cost;
      }
      ```
    - **Why It Works**: Greedily adds the lowest-weight edge that doesn't form a cycle using DSU.

---

## 🎨 **Graph Coloring**
- **Category**: Pattern
- **Tags**: #graph, #coloring, #bipartite, #greedy
- **Similar To**: DFS, BFS
- **Core Idea**: Assign colors to graph nodes such that no two adjacent nodes share the same color.
- **Time Complexity**: O(V + E)
- **Space Complexity**: O(V)
- **Limitations**: NP-complete for >2 colors
- **Use Cases**: Register allocation, scheduling problems
- **Example**:
    - **Title**: Bipartite Check using Coloring
    - **Code**:
      ```cpp
      bool isBipartite(vector<vector<int>> &graph) {
          int n = graph.size();
          vector<int> color(n, -1);

          for (int i = 0; i < n; ++i) {
              if (color[i] != -1) continue;
              queue<int> q;
              q.push(i);
              color[i] = 0;

              while (!q.empty()) {
                  int node = q.front(); q.pop();
                  for (int nei : graph[node]) {
                      if (color[nei] == -1) {
                          color[nei] = 1 - color[node];
                          q.push(nei);
                      } else if (color[nei] == color[node]) {
                          return false;
                      }
                  }
              }
          }
          return true;
      }
      ```
    - **Why It Works**: BFS tries to color each connected component in two colors.

---

## 🔍 **Graph Tarjan**
- **Category**: Algorithm
- **Tags**: #graph, #dfs, #strongly_connected_components, #bridge
- **Similar To**: Kosaraju
- **Core Idea**: DFS-based algorithm to find strongly connected components or bridges in a graph.
- **Time Complexity**: O(V + E)
- **Space Complexity**: O(V)
- **Limitations**: Recursive stack overflow for large graphs
- **Use Cases**: Circuit analysis, compilers, dependency resolution
- **Example**:
    - **Title**: Tarjan's Algorithm for SCC
    - **Code**:
      ```cpp
      void tarjanSCC(int u, int &time, vector<int> &low, vector<int> &disc, vector<bool> &inStack, stack<int> &st, vector<vector<int>> &sccs, vector<vector<int>> &graph) {
          disc[u] = low[u] = time++;
          st.push(u); inStack[u] = true;

          for (int v : graph[u]) {
              if (disc[v] == -1) {
                  tarjanSCC(v, time, low, disc, inStack, st, sccs, graph);
                  low[u] = min(low[u], low[v]);
              } else if (inStack[v]) {
                  low[u] = min(low[u], disc[v]);
              }
          }

          if (low[u] == disc[u]) {
              vector<int> scc;
              while (true) {
                  int v = st.top(); st.pop();
                  inStack[v] = false;
                  scc.push_back(v);
                  if (v == u) break;
              }
              sccs.push_back(scc);
          }
      }
      ```
    - **Why It Works**: Uses timestamps and low-link values to identify SCCs.

---

## 📚 **Heap**
- **Category**: Data Structure
- **Tags**: #heap, #priority_queue
- **Similar To**: Balanced BST, Priority Queue
- **Core Idea**: A Heap is a special tree-based data structure that satisfies the heap property: in a max-heap, each parent node is greater than or equal to its children; in a min-heap, each parent is less than or equal to its children.
- **Time Complexity**: Insert: O(log n), Remove: O(log n), Access Top: O(1)
- **Space Complexity**: O(n)
- **Limitations**: Not suitable for searching arbitrary elements.
- **Use Cases**: Priority queues, scheduling, Dijkstra's algorithm.
- **Example**:
    - **Title**: Min Heap using Priority Queue
    - **Code**:
      ```cpp
      priority_queue<int, vector<int>, greater<int>> minHeap;
      minHeap.push(5);
      minHeap.push(1);
      minHeap.push(10);
      cout << minHeap.top(); // Outputs 1
      ```
    - **Why It Works**: The `greater<int>` comparator ensures the smallest element is at the top.

---

## 📦 **Heap Top K**
- **Category**: Pattern
- **Tags**: #heap, #topk, #sorting
- **Similar To**: QuickSelect, Bucket Sort
- **Core Idea**: Use a min-heap to maintain the top K largest (or smallest) elements seen so far.
- **Time Complexity**: O(n log k)
- **Space Complexity**: O(k)
- **Limitations**: Less optimal than QuickSelect in average-case for fixed-k.
- **Use Cases**: Top K frequent elements, Top K largest numbers.
- **Example**:
    - **Title**: Top K Largest Elements
    - **Code**:
      ```cpp
      vector<int> topK(vector<int>& nums, int k) {
          priority_queue<int, vector<int>, greater<int>> minHeap;
          for (int num : nums) {
              minHeap.push(num);
              if (minHeap.size() > k) minHeap.pop();
          }
          vector<int> result;
          while (!minHeap.empty()) {
              result.push_back(minHeap.top());
              minHeap.pop();
          }
          return result;
      }
      ```
    - **Why It Works**: The heap always maintains the k largest numbers by discarding smaller ones.

---

## ⚖️ **Heap Two Heaps**
- **Category**: Pattern
- **Tags**: #heap, #median
- **Similar To**: Sliding Window Median
- **Core Idea**: Use two heaps (a max-heap and a min-heap) to balance lower and upper halves of data for quick median finding.
- **Time Complexity**: O(log n) per insertion
- **Space Complexity**: O(n)
- **Limitations**: Needs extra logic for balancing and edge case handling.
- **Use Cases**: Running median, sliding window median.
- **Example**:
    - **Title**: Find Median from Data Stream
    - **Code**:
      ```cpp
      priority_queue<int> maxHeap; // lower half
      priority_queue<int, vector<int>, greater<int>> minHeap; // upper half

      void addNum(int num) {
          maxHeap.push(num);
          minHeap.push(maxHeap.top());
          maxHeap.pop();

          if (minHeap.size() > maxHeap.size()) {
              maxHeap.push(minHeap.top());
              minHeap.pop();
          }
      }

      double findMedian() {
          if (maxHeap.size() == minHeap.size())
              return (maxHeap.top() + minHeap.top()) / 2.0;
          return maxHeap.top();
      }
      ```
    - **Why It Works**: Max-heap and min-heap allow quick access to the middle values.

---

## 🔗 **Heap Merge K Sorted**
- **Category**: Algorithm
- **Tags**: #heap, #merge, #divide_and_conquer
- **Similar To**: Merge Sort, K-way Merge
- **Core Idea**: Merge k sorted arrays/lists using a min-heap to always get the next smallest element.
- **Time Complexity**: O(n log k), where n is total elements.
- **Space Complexity**: O(k)
- **Limitations**: Overhead of maintaining heap structure.
- **Use Cases**: External sorting, merging data streams.
- **Example**:
    - **Title**: Merge K Sorted Lists
    - **Code**:
      ```cpp
      struct NodeCompare {
          bool operator()(ListNode* a, ListNode* b) {
              return a->val > b->val;
          }
      };

      ListNode* mergeKLists(vector<ListNode*>& lists) {
          priority_queue<ListNode*, vector<ListNode*>, NodeCompare> pq;
          for (auto list : lists)
              if (list) pq.push(list);

          ListNode dummy(0), *tail = &dummy;
          while (!pq.empty()) {
              ListNode* node = pq.top(); pq.pop();
              tail->next = node;
              tail = tail->next;
              if (node->next) pq.push(node->next);
          }
          return dummy.next;
      }
      ```
    - **Why It Works**: The heap always picks the smallest current head, efficiently merging all lists.

---

## 💡 **Bit Manipulation**
- **Category**: Technique
- **Tags**: #bit, #math, #optimization
- **Similar To**: Math Tricks
- **Core Idea**: Use bitwise operators (&, |, ^, <<, >>) to solve problems efficiently using binary representation.
- **Time Complexity**: O(1) to O(log n)
- **Space Complexity**: O(1)
- **Limitations**: Hard to debug and understand; edge cases with negatives.
- **Use Cases**: Subset generation, parity check, toggling bits.
- **Example**:
    - **Title**: Find Unique Element (Every element appears twice except one)
    - **Code**:
      ```cpp
      int singleNumber(vector<int>& nums) {
          int result = 0;
          for (int num : nums) result ^= num;
          return result;
      }
      ```
    - **Why It Works**: XOR cancels out duplicates, leaving the unique number.

---

## ➗ **Math**
- **Category**: Technique
- **Tags**: #math, #gcd, #modulo, #combinatorics
- **Similar To**: Number Theory
- **Core Idea**: Use mathematical concepts like GCD, LCM, modulo arithmetic, primes, factorials, etc., to solve problems.
- **Time Complexity**: Varies by operation
- **Space Complexity**: O(1) to O(n)
- **Limitations**: Integer overflow, requires deep understanding of theory.
- **Use Cases**: GCD, modular inverse, counting problems.
- **Example**:
    - **Title**: Compute GCD
    - **Code**:
      ```cpp
      int gcd(int a, int b) {
          return b == 0 ? a : gcd(b, a % b);
      }
      ```
    - **Why It Works**: Based on Euclidean algorithm which reduces the problem size quickly.

---

## 🎮 **Simulation**
- **Category**: Pattern
- **Tags**: #simulation, #brute_force
- **Similar To**: Backtracking, Greedy
- **Core Idea**: Mimic the process step-by-step as described in the problem, often involving state management.
- **Time Complexity**: Varies (usually high)
- **Space Complexity**: Varies
- **Limitations**: Inefficient for large inputs; can be slow.
- **Use Cases**: Game mechanics, state machines, robot movement.
- **Example**:
    - **Title**: Robot Returns to Origin
    - **Code**:
      ```cpp
      bool judgeCircle(string moves) {
          int x = 0, y = 0;
          for (char move : moves) {
              if (move == 'U') y++;
              else if (move == 'D') y--;
              else if (move == 'L') x--;
              else if (move == 'R') x++;
          }
          return x == 0 && y == 0;
      }
      ```
    - **Why It Works**: Simulates movement on a 2D plane and checks if it returns to the origin.




