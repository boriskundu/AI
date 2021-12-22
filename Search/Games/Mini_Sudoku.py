# ## Mini Sudoku of Dimension: 3 by 3 (Tree Search – Backtracking Algorithm – Recursive Implementation)

# ### Check if Board is valid
# Repeated values in rows, columns are not acceptable.
def board_valid(brd):
  for row in brd:
    used=[]
    for element in row:
      if element!=0 and element in used:
        return False
      else:
        used.append(element)
  for col in zip(*brd):
    used=[]
    for element in col:
      if element!=0 and element in used:
        return False
      else:
        used.append(element)
  return True


# ### Check if Board is filled
def board_filled(brd):
  for row in brd:
    for element in row:
      if element==0:
        return False
  return True


# ### Method to solve board
def solver(brd):
  if board_filled(brd):
    if board_valid(brd):
      return (brd)
    else:
      return None
  else:
    for i in [0,1,2]:
      for j in [0,1,2]:
        if brd[i][j]==0:
          for test in [1,2,3]:
            brd[i][j]=test
            if board_valid(brd) and solver(brd)!=None:
                return (brd)
            else:
              brd[i][j]=0


# Run with sample board
# ### Create board

Board = [ [ 0 , 0 , 0 ],
          [ 1 , 0 , 0 ],
          [ 0 , 3 , 1 ] ]
print(f'Initial Board:')
for item in list(zip(*Board)):
  print (item)

solution = solver(Board)
print('\nSolution:')
for row in solution:
  print (row)