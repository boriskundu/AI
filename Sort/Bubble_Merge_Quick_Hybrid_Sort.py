"""
Bubble, Merge, Quick and Hybrid Sorts

Author: BORIS KUNDU

"""

class Sort:
#----------------------------------------------------------
#                   1: Bubble Sort (Basic)
#----------------------------------------------------------
    # Function bubbleSort takes an unsorted list of numbers as input
    def bubbleSort(inputNumList):
        inputLength = len(inputNumList)
        # Check if we should attempt sorting
        if(inputLength > 1):
            # Outer loop
            for i in range(inputLength):
                # Inner loop
                for j in range(inputLength-1,0,-1):
                    # Sort in ascending order
                    if inputNumList[j] < inputNumList[j-1]:
                        # Swap values using temp
                        temp = inputNumList[j]
                        inputNumList[j] = inputNumList[j-1]
                        inputNumList[j-1] = temp
        # return sorted list
        return inputNumList


#----------------------------------------------------------
#                   2: Merge Sort (Basic)
#----------------------------------------------------------
    # Function mergeSort takes an unsorted list of numbers as input
    def mergeSort(inputNumList):
        inputLength = len(inputNumList)
        # Check if we should attempt sorting
        if(inputLength > 1):
            # Get middle index
            middle = inputLength//2
            # Define left half of unsorted list
            left = inputNumList[:middle]
            # Define right half of unsorted list
            right = inputNumList[middle:]
            # Sort left half recursively
            Sort.mergeSort(left)
            # Sort right half recursively
            Sort.mergeSort(right)
            # Define variables for iteration, comparison and merging 
            a=b=c=0
            left_size = len(left)
            right_size = len(right)
        # first loop to compare both halves, merge and sort in ascending order
            while a < left_size and b < right_size:
                if left[a] <= right[b]:
                    inputNumList[c]=left[a]
                    a=a+1
                else:
                    inputNumList[c]=right[b]
                    b=b+1
                c=c+1
            # second loop for left half merge
            while a < left_size:
                inputNumList[c]=left[a]
                a=a+1
                c=c+1
            # third loop for right half merge
            while b < right_size:
                inputNumList[c]=right[b]
                b=b+1
                c=c+1
        # return sorted list
        return inputNumList


#----------------------------------------------------------
#                  3: Quick Sort (Basic)
#----------------------------------------------------------
    # Function quickSort takes an unsorted list of numbers as input
    def quickSort(inputNumList):
        inputLength = len(inputNumList)
        # Check if we should attempt sorting
        if(inputLength > 1):
            # Set initial positions
            start = 0
            end = inputLength-1
            # Call recursive function
            Sort.quickRecur(inputNumList,start,end)
        # return sorted list
        return inputNumList

    def quickRecur(inputNumList,start,end):
        if start < end:
                # find split point
                middle = Sort.divide(inputNumList,start,end)
                # process one half
                Sort.quickRecur(inputNumList,start,middle-1)
                # process other half
                Sort.quickRecur(inputNumList,middle+1,end)

    def divide(inputNumList,start,end):
        # Define pivot value
        pivot = inputNumList[start]
        # Set left and right positions
        left = start+1
        right = end
        processed = False
        while not processed:
            while left <= right and inputNumList[left] <= pivot:
                left = left + 1
            while inputNumList[right] >= pivot and right >= left:
                right = right - 1
            if right < left:
                processed = True
            else:
                # Swap values using temp
                temp = inputNumList[left]
                inputNumList[left] = inputNumList[right]
                inputNumList[right] = temp
        # Swap values using temp
        temp = inputNumList[start]
        inputNumList[start] = inputNumList[right]
        inputNumList[right] = temp

        return right

#----------------------------------------------------------
#          4: Hybrid Sort (both quick and merge)
#----------------------------------------------------------
    #Function to read input and apply appropriate sort type
    def HybridSort(L, BIG, SMALL, T, SORTED):
        BIG = BIG.lower()
        SMALL = SMALL.lower()

        if (len(L)<=T):
            SORTED = Sort.bubbleSort(L)
            print(' {:>20} : Small - {}'.format(' Hybrid Style used',SMALL))

        elif (BIG == 'mergesort'):
            SORTED = Sort.HybridMergeSort(L,T) #Defined below
            print(' {:>20} : Big - {}'.format(' Hybrid Style used',BIG))

        elif (BIG == 'quicksort'):
            SORTED = Sort.HybridQuickSort(L,T) #Defined below
            print(' {:>20} : Big - {}'.format(' Hybrid Style used',BIG))

        return (SORTED)
    
    #----------------------------------------------------------
    #                   4(a): Hybrid Merge Sort
    #----------------------------------------------------------
    """ 
        1. Split the list until the size of the list reaches threshold value using Merge sort principles
        2. Upon reaching threshold, use Bubble sort to sort the sub-list
        3. Combine the sublists using merge sort principles to present final output
    """
    def HybridMergeSort(L,T):
        len_of_list = len(L)
        if(len_of_list <= T):
            mid = len_of_list//2
            #Recurse on left and right halves of the list to reach threshold
            Sort.HybridMergeSort(L[:mid],T)
            Sort.HybridMergeSort(L[mid+1:],T)
        #List size below threshold, use Bubble sort
        return Sort.bubbleSort(L)
 
    #----------------------------------------------------------
    #                   4(b): Hybrid Quick Sort
    #----------------------------------------------------------
    """
        1. Split the list until its size reaches threshold using the Quick sort partitioning technique 
        2. Upon reaching threshold, use Bubble sort to sort the sub-list
        3. Combine the sublists using quick sort principles and present the output
    """
    def Partitioning_QuickSort(L, start, end):
        pivot_idx = end
        pivot = L[pivot_idx] # Last element as Pivot
 
        smaller_element = -1
        for i in range(0, pivot_idx):
            if (L[i] < pivot):
                smaller_element+=1
                L[smaller_element], L[i] = L[i], L[smaller_element]     
        #Place the pivot element in its correct location for splitting
        L[smaller_element+1], L[pivot_idx] = L[pivot_idx], L[smaller_element+1]
        #Return the pivot element to main function
        return (smaller_element + 1)
    
    #Function for Hybrid Quick sort recursion 
    def HybridQuickSort(L, T):
        len_of_list = len(L)
        if(len_of_list > T):
            #Get the index of the pivot element to split the list
            pivot_index = Sort.Partitioning_QuickSort(L, 0, len_of_list-1)
            #Split the list using pivot and recurse
            Sort.HybridQuickSort(L[:pivot_index-1], T)
            Sort.HybridQuickSort(L[pivot_index+1:], T)
        
        return Sort.bubbleSort(L)

"""Execution Test"""

# TEST FUNCTION
# Test the three basic versions of Bubble, Quick and Merge sort algothims.
def TestBasicSort(id, L):

    print('<<< TestID: {} - Input Parameters: >>>\n List \t: {} \n'.format(id, L))
    print(' {:>20} : {}'.format('Bubble Sort output', Sort.bubbleSort(L)))
    print(' {:>20} : {}'.format('Merge Sort output', Sort.mergeSort(L)))
    print(' {0:>20} : {1}'.format('Quick Sort output', Sort.quickSort(L)))
    print('')

# Test the three hybrid versions of Quick+Bubble and Merge+Bubble sort algothims.
def TestHybridSort(id, L, BIG, SMALL, T):
    SORTED = []
    print('<<< TestID: {} - Input Parameters: >>>\nList\t: {} \nBIG\t: {}\nSMALL\t: {}\nT\t: {}\n'.format(id, L,BIG,SMALL,T))
    print(' {0:>20} : {1}'.format('Hybrid Sort output', Sort.HybridSort(L, BIG, SMALL, T, SORTED)))
    print('')

# TEST CASES
# Define unsorted input lists for sorting
L1 = [5,1,7,3,0]
L2 = [51,12,73,37,3,88,45,29,91,67]
L3 = [-5,10,70,-3,0,-81,42,99,-65]
L4 = [-51,100,170,-3,30,-81,42,291,99,77]
L5 = [-511,102,175,-23,310,-81,142,291,199,-65,211,45,0,91]

TestBasicSort(1, L1)
#OUTPUT
""" 
<<< TestID: 1 - Input Parameters: >>>
 List   : [5, 1, 7, 3, 0]

   Bubble Sort output : [0, 1, 3, 5, 7]
    Merge Sort output : [0, 1, 3, 5, 7]
    Quick Sort output : [0, 1, 3, 5, 7]

"""
TestHybridSort(2, L2, 'mergesort', 'bubbleSort', 10)
#OUTPUT
""" 
<<< TestID: 2 - Input Parameters: >>>
List    : [51, 12, 73, 37, 3, 88, 45, 29, 91, 67]
BIG     : mergesort
SMALL   : bubbleSort
T       : 10

    Hybrid Style used : Small - bubblesort
   Hybrid Sort output : [3, 12, 29, 37, 45, 51, 67, 73, 88, 91]

"""
TestHybridSort(3, L3, 'quicksort', 'bubbleSort',10)
#OUTPUT
""" 
<<< TestID: 3 - Input Parameters: >>>
List    : [-5, 10, 70, -3, 0, -81, 42, 99, -65]
BIG     : quicksort
SMALL   : bubbleSort
T       : 10

    Hybrid Style used : Small - bubblesort
   Hybrid Sort output : [-81, -65, -5, -3, 0, 10, 42, 70, 99]

"""
TestHybridSort(4, L4, 'quicksort', 'bubbleSort', 5)
#OUTPUT
"""
<<< TestID: 4 - Input Parameters: >>>
List    : [-51, 100, 170, -3, 30, -81, 42, 291, 99, 77]
BIG     : quicksort
SMALL   : bubbleSort
T       : 5

    Hybrid Style used : Big - quicksort
   Hybrid Sort output : [-81, -51, -3, 30, 42, 77, 99, 100, 170, 291]

"""
TestHybridSort(5, L5, 'mergesort', 'bubbleSort', 5)
#OUTPUT
"""
<<< TestID: 5 - Input Parameters: >>>
List    : [-511, 102, 175, -23, 310, -81, 142, 291, 199, -65, 211, 45, 0, 91]
BIG     : mergesort
SMALL   : bubbleSort
T       : 5

    Hybrid Style used : Big - mergesort
   Hybrid Sort output : [-511, -81, -65, -23, 0, 45, 91, 102, 142, 175, 199, 211, 291, 310]

"""