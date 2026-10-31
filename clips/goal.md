# Ground-truth goals

Marker: 9.5 cm, DICT_4X4_50 id 0.
Goal types: put_all, move, arrange, stack, swap.

## c01
goal: put_all(pencil, cup)
scope: all
order_matters: false
objects: black pencil, orange pencil, brown pencil, white cup, blue eraser, black earphone
moved: brown pencil, orange pencil, black pencil
narration: "I'm going to put the pencils in the cup."
notes: eraser and earphone are distractors, never touched

## c02
goal: arrange(objects, row)
scope: all
order_matters: false
objects: orange pencil, purple marker, blue eraser, black earphone
moved: orange pencil, purple marker, blue eraser, black earphone
narration: "Now I am going to arrange each object to align with eachother"
notes:

## c03
goal: stack(small eraser, big eraser)
scope: this
order_matters: true
objects: small eraser, big eraser
moved: small eraser
narration: "Now I'm gonna stack the small eraser on the big eraser."
notes:

## c04
goal: move(black wallet, next_to, pencil)
scope: this
order_matters: false
objects: black wallet, silver keychain, pencil
moved: black wallet, silver keychain
narration: "I'm moving my wallet next to my pencil."
notes: keychain attached to wallet, should be grouped

## c05
goal: swap(scissor, eye glasses)
scope: this
order_matters: false
objects: scissor, eye glasses
moved: scissor, eye glasses
narration: "I'm swapping the position of the scissor with the position of the eye glasses."
notes:

## c06
goal: put_all(purple marker, white scissor, brown pencil lead case, blue eraser, black pencil)
scope: this
order_matters: false
objects: purple marker, white scissor, brown pencil lead case, blue eraser, black pencil, white cup
moved: purple marker, white scissor, brown pencil lead case, blue eraser, black pencil
narration: "In this task here we are putting 5 object into the cup starting of with the marker, then the scissor, then the pencil lead case, and the eraser, and the pencil"
notes:

## c07
goal: move(blue eraser)
scope: this
order_matters: false
objects: purple marker, white remote, white scissor, blue eraser, black pencil
moved: blue eraser
narration: "I will be moving the eraser to the left most position"
notes:

## c08
goal: move(black wallet)
scope: this
order_matters: false
objects: silver keychain, white remote, blue eraser, black pencil, black wallet
moved: black wallet
narration: "I'm sliding the wallet to the left"
notes: keychain attached to wallet, should be grouped

## c09
goal: move(black pencil)
scope: this
order_matters: false
objects: silver keychain, white remote, blue eraser, black pencil, black wallet
moved: black pencil
narration: "I'm moving the pencil over here"
notes: keychain attached to wallet, should be grouped


## c10
goal: put all(pencil, cup)
scope: this
order_matters: false
objects: black pencil, cream pencil, red pencil, white cup
moved: black pencil, cream pencil, red pencil
narration: "I'm moving all the pencils into the cup"
notes: