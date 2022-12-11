# Neetcode

## Top k frequent Elements

Solution in C++:

```{C++}
class Solution {
public:
    typedef pair<int,int> intPair;
    vector<int> topKFrequent(vector<int>& nums, int k) {
        priority_queue<intPair, vector<intPair>, less<intPair>> pq;
        
        unordered_map<int,int> freq;
        for(int n: nums){
           freq[n]++; 
        }
        
        for(auto &[n,f]: freq) {
            pq.push({f,n});
        }
        
        vector<int> res;
        for(int i = 0; i < k; i++){
           res.push_back(pq.top().second);
           pq.pop(); 
        }
        
        return res;
    }
};	
```

Notes:

- priority_queue<Key, Container, Compare>
- Use std::less for max heap 
- Use std::greater for min heap
- priority_queue of pairs sorted by first value



Link: https://leetcode.com/problems/top-k-frequent-elements



## Product of Array Except Self

Note: Make two passes of array

```{C++}
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        vector<int> res(nums.size(), 1);
        
        vector<int> prefix(nums.size(), 1);
        vector<int> postfix(nums.size(), 1);
        
        for(int i = 0; i < nums.size(); i++) {
            prefix[i] *= nums[i];
            if(i > 0) prefix[i] *= prefix[i-1];
        }
        
        for(int i = nums.size()-1; i >= 0; i--) {
            postfix[i] *= nums[i];
            if(i+1 < nums.size()) postfix[i] *= postfix[i+1];
        }
        
        for(int i = 0; i < nums.size(); i++) {
           if(i > 0) res[i] *= prefix[i-1];
           if(i+1< nums.size()) res[i] *= postfix[i+1];
        }
        
        return res;
    }
};
```





## Trapping Rain Water

Dyammic Programming



```python
class Solution:
    def trap(self, height: List[int]) -> int:
        if len(height) == 0:
            return 0
        
        ans = 0
        
        n = len(height)
        
        left_max = [0] * n
        right_max = [0] * n
        
        left_max[0] = height[0]
        right_max[n-1] = height[n-1]
        
        for i in range(1, n):
            left_max[i] = max(left_max[i-1], height[i])
        
        for i in range(n-2,-1,-1):
            right_max[i] = max(right_max[i+1], height[i])
        
        print(left_max, right_max)
        
        for i in range(n):
            ans += min(left_max[i], right_max[i]) - height[i]
        
        return ans
```



# Linked Lists

## Merge Two Sorted Lists

```{C++}
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        if(!list1) { return list2; }
        if(!list2) { return list1; }
        
        if(list1->val <= list2->val) {
            list1->next = mergeTwoLists(list1->next,list2);
            return list1;
        } else {
            list2->next = mergeTwoLists(list1,list2->next);
            return list2;
        }
    }
};
```



# Graphs

## Minimum cost to connect all points

* Kruskal's algorithm
* Minimum spanning tree
* Implementation of disjoint set with union by size and path compression
* heapq python interface for priority queue
  * element := (rank, data)
  * heappop(queue)
  * heappush(queue, element)

Pseducode for Kruskal's Algorithm:

```
1. Sort the edges in increasing order of weights
   (You can do this with a heap, or simply sort them in an array)
2. Initialize a separate disjoint set for each vertex
3. for each edge uv in sorted order:
4.   if u and v are in different sets:
5.     add uv to solution
6.     union the sets that u and v belong to
```

Code:

```{python}
import heapq
from typing import List


class Solution:
    # Kruskal's Algorithm
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        n = len(points)
        dset = [-1] * n
        pq = []
        cost = 0

        def find(x: int):
            assert x < n

            if dset[x] < 0:
                return x

            dset[x] = find(dset[x])
            return dset[x]

        def union(x: int, y: int):
            assert x < n
            assert y < n

            xroot = find(x)
            yroot = find(y)

            if xroot == yroot:
                return False

            newsize = dset[xroot] + dset[yroot]

            # union by size
            if xroot < yroot:
                dset[yroot] = xroot
                dset[xroot] = newsize
            else:
                dset[xroot] = yroot
                dset[yroot] = newsize

            return True

        def manhattan(i, j):
            return abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])

        for i in range(n):
            for j in range(i + 1, n):
                heapq.heappush(pq, (manhattan(i, j), (i, j)))

        while pq:
            dist, (i, j) = heapq.heappop(pq)
            if union(i, j):
                cost += dist

        return cost


s = Solution()
print(s.minCostConnectPoints([[0, 0], [2, 2], [3, 10], [5, 2], [7, 0]]))
```



## Network Delay Time

* Dijkstra's Algorithm
* Weighted Graph
* Nodes are numbered from 1 to n
* The graph is represented as a map u -> [(v1, w1), (v2, w2), ...] where u is the src node, v is the dest node, w is the edge weight or "cost"

```{python}
from typing import List
from heapq import heappush, heappop
from collections import defaultdict
import math

class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        if n < 1:
            return 0

        graph = defaultdict(list)
        visited = set()
        dist = {} # dist contains min time to reach each node from node k
        queue = [] # priority queue

        # init graph with adjacency list
        for u, v, w in times:
            graph[u].append((v, w))

        for i in range(1, n + 1):
            dist[i] = math.inf

        heappush(queue, (0, k))
        dist[k] = 0

        # dijkstra's algorithm
        while queue:
            _, u = heappop(queue)

            # duplicate entries allowed
            if u in visited:
                continue

            visited.add(u)

            for v, cost in graph[u]:
                if v not in visited:
                    if dist[u] + cost < dist[v]:
                        dist[v] = dist[u] + cost
                    heappush(queue, (dist[v], v))

        # check if all nodes are visited
        if len(visited) < n:
            return -1

        # min time to reach all nodes is the time to reach the slowest one
        return max(dist.values())

```



## Swim In Water

Dijkstra's Algorithm

```{python}
"""
You are given an n x n integer matrix grid where each value grid[i][j] represents the elevation at that point(i, j).

The rain starts to fall. At time t, the depth of the water everywhere is t. You can swim from a square to another 4 - directionally adjacent square if and only if the elevation of both squares individually are at most t. You can swim infinite distances in zero time. Of course, you must stay within the boundaries of the grid during your swim.

Return the least time until you can reach the bottom right square(n - 1, n - 1) if you start at the top left square(0, 0).
"""
import heapq


class Solution:
    def swimInWater(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """

        n = len(grid)
        q = []
        visited = set()
        dirs = [-1, 0, 1, 0, -1]

        ans = max(grid[0][0], grid[n - 1][n - 1])

        heapq.heappush(q, (0, 0, 0))

        while q:
            t, row, col = heapq.heappop(q)
            visited.add((row, col))
            ans = max(ans, t)

            if row == n - 1 and col == n - 1:
                return ans

            for i in range(4):
                r = row + dirs[i]
                c = col + dirs[i + 1]

                if r >= 0 and c >= 0 and r < n and c < n:
                    if (r, c) not in visited:
                        heapq.heappush(q, (grid[r][c], r, c))

        return -1

```

