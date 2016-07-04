# Rock Paper Scissors
# FSM Programming - June 6, 2016
# IU - Jaimie Murdock and Blake Forland
import random

choices = ['rock', 'paper', 'scissors']

def computer():
    "Returns a rindom chioce for rock paper scissors"
    return random.choice(choices)

def throw(prompt):
    """
    promopts the user for a vaild choice. If an invalid choice was made,
    continue to prompt until one was made. Returns the choice

    :param promt: The text to use for asking the user.
    :type prompt: string
    """
    answer = input(prompt)
    while answer not in choices:
        print("What was that?! Try again.")
        answer = input(prompt)
    
    return answer

def game():
    print("Rock. Paper. Scissors. Go!")
    player1 = throw("Your throw? ")
    player2 = computer()

    record = "Player 1: {}    Player 2: {}"
    print(record.format(player1, player2))

    if player1 == player2:
        print("It's a tie!")
        return
    if player1 == 'rock' and player2 == 'paper':
        print("Player 2 wins!")
        return 2
    if player1 == 'rock' and player2 == 'scissors':
        print("Player 1 wins!")
        return 1
    if player1 == 'paper' and player2 == 'rock':
        print("Player 1 wins!")
        return 1
    if player1 == 'paper' and player2 == 'scissors':
        print("Player 2 wins!")
        return 2
    if player1 == 'scissors' and player2 == 'rock':
        print("Player 2 wins!")
        return 2
    if player1 == 'scissors' and player2 == 'paper':
        print("Player 1 wins!")
        return 1

def tournament(rounds):
    p1wins = 0
    p2wins = 0
    match = 0
    
    while (p1wins + p2wins) < rounds:
        # print the match number
        match = match + 1
        print("Round {}".format(match))

        # play the game
        winner = game()
        if winner == 1:
            p1wins = p1wins + 1
        elif winner == 2:
            p2wins = p2wins + 1
        print("Player 1: " + str(p1wins) + " Player 2: " + str(p2wins))

        # short circuit if someone won
        if p1wins > (rounds / 2):
            return 1
        elif p2wins > (rounds / 2):
            return 2
