// 本题需要反转链表中 指定一截元素
// 考虑到题目要求只能遍历一遍，所以想到额外声明一些辅助指针，内部采用头插法进行反转，然后将反转后的列表接回去即可
// 因为感觉这道题主要考察的是指针的用法，所以并没有使用python来进行实现，而是选择C++来进行编程，具体代码如下

/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* reverseBetween(ListNode* head, int m, int n) {
        
        ListNode* m_head = NULL;
        ListNode* now = NULL;
        ListNode* til = NULL;
        ListNode* temp = NULL;
        now = head;
        
        if(head->next == NULL)
            return head;
        
        if(m != 1){
            for(int i = 1; i <= n; i++){
                if(i == m-1){
                    m_head = now;
                    now = now->next;
                }
                if(i == m){
                    til = now;
                    now = now->next;
                }
                if(i > m){
                    temp = now;
                    now = now->next;
                    temp->next = m_head->next;
                    m_head->next = temp;
                }
                if(i < m-1)
                    now = now->next;
            }

            til->next = now;
            return head;
        }
        else{
            m_head = new ListNode(1);
            m_head->next = head;
            for(int i = 1; i <= n; i++){
                if(i == m){
                    til = now;
                    now = now->next;
                }
                if(i > m){
                    temp = now;
                    now = now->next;
                    temp->next = m_head->next;
                    m_head->next = temp;
                }
                if(i < m-1)
                    now = now->next;
            }

            til->next = now;
            return m_head->next;
        }
        
    }
};
