# ## Tic-Tac-Toe (MIN-MAX)

#Evaluate board moves
def evaluate(brd):
    win_cases = [
        [brd[0][0], brd[0][1], brd[0][2]],
        [brd[1][0], brd[1][1], brd[1][2]],
        [brd[2][0], brd[2][1], brd[2][2]],
        [brd[0][0], brd[1][0], brd[2][0]],
        [brd[0][1], brd[1][1], brd[2][1]],
        [brd[0][2], brd[1][2], brd[2][2]],
        [brd[0][0], brd[1][1], brd[2][2]],
        [brd[2][0], brd[1][1], brd[0][2]],
    ]
    if ['X','X','X'] in win_cases:
      score=1
    elif ['O','O','O'] in win_cases:
      score=-1
    else:
      score = 0
    return score


#Choices 'X' and 'O' and '_' for empty
board = [
    ['_', '_', '_'],
    ['_', '_', '_'],
    ['_', '_', '_'],
]
print(f'Empty board:{board}')
print(f'Evaluate empty board:{evaluate(board)}')

board = [
    ['X', '_', '_'],
    ['_', 'X', '_'],
    ['_', '_', 'X'],
]
print(f'Winning move on board:{board}')
print(f'Evaluate winning move:{evaluate(board)}')

#Check for empty spots and choices

def spots2fill(brd):
    spots=[]
    for i in [0,1,2]:
      for j in [0,1,2]:
        if brd[i][j] == '_':
          spots.append([i, j])
    return spots

board = [
    ['X', '_', '_'],
    ['_', 'X', '_'],
    ['_', '_', 'X'],
]
print(f'Empty spots on board:{board}')
print(f'Spots to fill:{spots2fill(board)}')

# ### 4. Decision maker:

#a=[2,1,3]
#a.index(min(a))

#Min Max
def min_max(brd,player):
  if spots2fill(brd)==[] or evaluate(brd)==1 or evaluate(brd)==-1:
    return [None, None, evaluate(brd)]
  else:
    All_spot_score=[]
    All_spot_coordinates=[]
    for spot in spots2fill(brd):
      brd[spot[0]][spot[1]]=player
      if player=='X':
        spot_score=min_max(brd,'O')[2]
      else:
        spot_score=min_max(brd,'X')[2]
      brd[spot[0]][spot[1]]='_'
      All_spot_score.append(spot_score)
      All_spot_coordinates.append(spot)
    if player=='X':
      best_score=max(All_spot_score)
      best_score_index=All_spot_score.index(max(All_spot_score))
    else:
      best_score=min(All_spot_score)
      best_score_index=All_spot_score.index(min(All_spot_score))      
    return [All_spot_coordinates[best_score_index][0],All_spot_coordinates[best_score_index][1],best_score]

board = [
    ['O', 'O','X'],
    ['X', '_', 'O'],
    ['X', '_', 'X'],
]

print('Check for Min-Max')
for item in board:
  print (f'Item:{item}')
print(f"Min_Max with O:{min_max(board,'O')}")

board = [
    ['_', '_', '_'],
    ['_', '_', '_'],
    ['_', '_', '_'],
]

# Main loop of this game
while (len(spots2fill(board))>0 and evaluate(board)==0):
  moves = {1: [0, 0], 2: [0, 1], 3: [0, 2],
           4: [1, 0], 5: [1, 1], 6: [1, 2],
           7: [2, 0], 8: [2, 1], 9: [2, 2]}
  move = int(input('\nYour turn:\nEnter (1 ~ 9): '))
  if board[moves[move][0]][moves[move][1]]!='_':
    print ("\nSpot is already filled\n")
    continue
  board[moves[move][0]][moves[move][1]]='X'
  for item in board:
    print (item)
  result=min_max(board,'O')
  #print (result)
  if result[0]==None:
    continue
  else:
    print ("\nComputer Move: \n")
    board[result[0]][result[1]]='O'
    for item in board:
      print (item)
 
# Game over message
print("\nThe end:")
if evaluate(board)==1:
    print('YOU WON!')
elif evaluate(board)==-1:
    print('YOU LOST!')
else:
    print('DRAW!')