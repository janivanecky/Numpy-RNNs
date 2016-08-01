# Numpy-RNNs

To get a better grasp of recurrent networks, I decided to do something basic and cool and so I implemented character level language model using 
vanilla RNNs from scratch in numpy, inspired by <a href="https://gist.github.com/karpathy/d4dee566867f8291f086">this gist by Andrej 
Karpathy</a>. To make it more interesting, I also implemented support for multiple layers and dropout on output. There is a problem
with training vanilla RNNs though, vanishing and exploding gradients, making it hard for the network to learn long term dependencies. 
LSTMs are used to overcome this problem, but they add a bit more of complexity. We want to keep things simple, so some people came up with 
simple solutions to the problem of learning long term dependencies in vanilla RNNs by using ReLU activations and clever intializations:

* IRNN by Le <i>et al.</i>(https://arxiv.org/pdf/1504.00941.pdf): Authors proposed using ReLU as an activation function instead fo Tanh and initializing recurrent
weights matrix to an identity matrix. This means that at the beggining, state is propagated to the next time step without any loss of information.

* NPRNN by Talathi <i>et al.</i>(http://arxiv.org/pdf/1511.03771v3.pdf): Similar approach as IRNN, but they propose using positive definite matrix with largest 
eigenvalue 1 as an initialization for recurrent weights. This should increase the stability of model compared to IRNN.

I decided to test out these propositions and modified the source code of vanilla RNN to obtain these four models:

* `tanh.py` - Vanilla RNN with a Tanh activation
* `relu.py` - Vanilla RNN with a ReLU activation
* `stirnn.py` - IRNN, but since the original implementation was highly unstable during training and I wasn't able to overcome this, I used 
solutions proposed by Krueger <i>et al.</i> (http://arxiv.org/pdf/1511.08400.pdf) and added stabilizing term to the loss function which enforces
norms of successive states to be of similar size.
* `nprnn.py` - NPRNN



## Results

To test the models, I trained all four models on <a href="https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare"><i>tinyshakespear</i></a> (thanks to <a href="https://twitter.com/karpathy">Andrej</a>) 
file that I split into training (shakespear_train.txt) and validation set (shakespear_val.txt). Each model has 3 layers, 256 hidden units in each layer 
and I added dropout with rate 0.1 to the outputs of each layer. 
I trained using RMSProp with a learning rate 0.01, batch size 10 and I backpropagated errors for 100 steps. Training ran for ~100 epochs.

![Training loss comparison](/graphs/train_all.png)
![Validation loss comparison](/graphs/val_all.png)

Ok, so first thing that's obvious, stabilized IRNN (STIRNN) doesn't work very well, probably because the stabilizing term constrains the network 
too much and prevents it from learning the task. This doesn't confirm the results presented in the paper, so it may just be a mistake in my implementation, or 
IRNN requires very delicate setting of hyperparameters, which isn't mentioned in the paper.
<br>
Other than that, we can conclude that using ReLU in simple RNNs works better then using Tanh activation and that NPRNNs work marginally 
better then simple RNNs. Now let's have a look at Shakespear imitations produced by the models: 

### Vanilla RNN with Tanh activation (`tanh.py`):
```
POMPEY:
She be the shall doth the pate the courness,
That lead by the sake not well to the cloud
she fecence of the cousin to be all their ather;
And mere you the more that time to did her by you
To see the pain of my breath make the earth and us.

CAMILLO:
What seeming all the blood be gone and would
But rememberly, say you at the resse?

KATHARINA:
I say thee and all men belourd in his cousur!

MENENIUS:
Perture in absemary on my poor love;
And prince it lady! where for the country!

GLOUCESTER:
I shall lay unterreators, but with thee was
ship the tages of my lord, you might heard
A this but our brother's dounish their ears!

Third Servant:
My lord, he respect the athershale well to you.

PADI CADULEO:
See he hath you gone and my lady's sard.

POMPEY:
The mostath for their queen's peace, I ar with her.

LUCIO:
Why there is ever for his heart will we be
I dead of you, for your borish'd false of his
come on thy words in that way to grant,
And all the world to make we will you be,
That revolt the conceing my hate as I that looks
And so we shall be so but one for the grief;
For what is in the sear a score to the love
And batience of the rable to hear you:
But then him is sow and shall know thee an upon my
even here is no more to see thee long and have
To came unhright that you do love be obe
To stores.

ANGELO:
That sink all for the grace of such armost;
And comes the queen's leave my lord;
For that you were land of this word, you that
I have heard as he in this purised,
So great a are to grave and good with his recount
Unther is the stranger, thy engean your hands
And make my tage your son, sir, faith, never
Because with the breath of man the faress to me of the
constain and the gods of him that is recome.

GLOUCESTER:
Heaven to the mantle light, love the blood,
To chance where and his as thy pregeret's repoltage:
For he is come to be so do not become
And gentle and for his ray with seep
By the death is your love to not so died,
To be patient walls
```

### Vanilla RNN with ReLU activation (`relu.py`):

```
LADY ANNE:
A will that your ready, night is end.

BAPTISTA:
Take her you at the grave gone:
The earth are come and love thouse a banish'd?

LUCIO:

Provost:
Not help, we were not we would swear my death:
Say, in the sign wost to it as one day me door nothing,
I would the husband at as the youth,
And unstanctes this Edward of him.

YORK:
Why, nay, therefore we cain so gate to the more
which I have friend comfort of home and the earth.

KING RICHARD III:
She earth he lover by thy heart.

PETRUCHIO:
And you will live my sight than this time of the seise to land
I'll be for the body man
on the did with bettle bail! battles and me to so denied
And bring mercy; sir, he as not 'twas a lords.

HENRY BOLINGBROKE:
Thy constrate after the infection to a give to have
As thou destruction be she way as the seem and
prosperous shoulder and becal the duke as entertion,
That is a father and some boals.

LUCIO:
Why, how say you have made a bowd to stay,
This is the power than our sons; but I is the son the people,
For when I have say you are he hath saw
As that lamper secursty that too my life;
May and but one shamb that banish'd place and save,
And the should of his eyes. I to the duke and now.

POMPEY:
You shall be a time: I see it for the full to the parts
You did by Perpechion to take it him and to
make his grace.

Post:
I should be thou did with A man was then;
Then what she will sigh that silken haste,
I'll make me to shall be she garks and then,
Your voices arm and lead in the world,
A lad of the uster he stand done many
Turns of brother's provess the desperation.

DUKE VINCENTIO:
Ole makes your kind and reled and the reason
That I had dead this been to envellate,
Hortensio, which hast thou apent
That his torture on me be say.

SICINIUS:
I live death are presure.
```

### IRNN (`stirnn.py`):

```
:  e   e anyoe n r ,Osept,h Of m:rsh  
 ilodd y hS bb n o
t ruhs  eom aaolo,e:hipeGrt od o otrtwy l l
hekiheroa r t  sedu eaola n t sn
n  e neo nn  wra
dqml er
 yy son e  teiye
rlps  elwsa  noilne:: t  anr 
so o h:oae ote tehatu rl!fnu hHa o h   r   eo ue ee o i  sdic rt  a r isw l ltyaa respfi,i e  n, n.u
oaloelrt edhoiem ere d WAeotek yse h
till :nrad hi ewer
h  iTn hees ao sotOT ralite    eiis  fi aHreuabhSoheh 
omb  afeb o
lo ihnr he  a  rg efy i    saeytasCfhhfo thtsp
 oaO  u$    s doaanheeae, NMo Eyel cra
pt  h, ced   sA saeo;h:a eoi  ah   s  ro rtrsa  Der   net teit  saa,re hh heos;  aeton  :
r nueKo rthtie,at,ts e sIs lws Poihm s  rthe f a Qjtu
o t   blto
dte wy  teg iomne  wehE ilne eek  b nra
V eo sleopsyee ep r gonoi hega haaea  lh o
si
lah  ed 
ar    hel doItesay 
 rOltbeisvI
l mK L
 nrntn hhtnaet ae firlced ah vset
nf  awaurnh  ebo Ctltn  rf h o e
     thes 
dw o  hfnotd sd  efeu  shtyybolmeo  ssh oedi  ?  ni  tand  aai l e
 srfgNnteh r ntdeeeU A eoe s  
r o o et meatto  eati:il lf  edbsi nhuet :ologt'lEwh, tsi  tiees hio t a elot nlvnei
  vIirbdh  rohfe    g tdste
tr o  rjhohae  uaoaha kae g doeeny yoF:uy ss,eaed tu u en ots edsiAhSr sn  os  i nT et v,hekero ttI r kio aA   e  o tteralta.htni
d  htle
eai
mst  yeoil iPoh?,so eof ane e 
tlO udwt  io g eosca 
rn e  t N  os aod ti: i
oatd  
to
 hd r  Ttofh  oimr tt:te oe ne el 
m;dyeu s   ht i 
Assso hhn ecE toolhhi  ot   hamn  
sae,eEadnn nltc
net l  o
  ete, s  ed'lt oera r:swnteea wh  oryuWrh      yrohm  ah
l  ae
e

ko s ew uLar

  n ode
t  a
he no l osvsso sd  S o eL heae 
k  u     reil   n t   tosesEhm es ha d t e  htwe eoA oee   h drnn
eet u
setnutroIoe rois ehkoiueiBn  uiou o edteIaioy
f
Isseaoa mlwo at  EI uai ne ttoe rrso
he,aeded
antnosrt adR
EI   ye
nlg ehs 
wlin:sha  w so,sraaN M 
eie,ehsc so r ep:aln nr  ee cerfendAo 
hifhthshate
cLe 
h  n en aii i his  n  t n&sno tr u ai unn
n
eehr otht irad's trato
edansa   f i
hon  hemeet e  eelu se ts laeE 
ril'o  toeg   tes n   mdie, e to
sait ot eo
 e'a e
```

### NPRNN (`nprnn.py`):

```
BUCKINGHAM:
A man I all their like by overtone after
Lucention and with I fear his single had
doubt being this for the monering, and roye.

AUTOLYCUS:
No, my lord, to that it come!

SEBASTIAN:
I pray you, love and care should on thee?

BIONDELLO:
'Tis look his place gentleman. But the farewell:
Bride we presently villain'd by your weary forth.
And from the mustice there of itself one hath
O hand of the woman and made with that say show
The mistress of the life and not and pleased.

NORTHUMBERLAND:
Bet you have stop the world is to be king
The rest man see the friends.

Second Citizen:
Why, he is your most die and fashion of the bed
In that a course as that we will not up:
Thou dignition begle her hindered for the wire:

First Senator:
I come, let him not put of their since of her shall
More than when the death and right the Death, Marcius
Tase our cousin, that bay-sender the world!

GLOUCESTER:
If he shall peal him. Thy war, and her heaven,
That I will falf wretling fleed and faw,
And nothing fall of him to a connot thalk
That I am, and yet to hear them speak?

GROMIO:
Thou art a come to be the fatheries
Was to my lord. How do to whom to have,
What women soul be more best, thou hast to him too
The full to reselve them to this brother vacking
forth the vensue of all as I do did him
To tell you are thee, for you are he peace,
Were bloods are leperge of first that will thou blind and this
as the name of your holy and mine from the gates;
And the play to a callence long broken, Secries.

TINBAS:
Hawh shall be content in the fortuness
Which death and seal must be dially deed:
To clear upon the present and such a tribpy,
My consil of all the fortune to let his crood,
And that the bold, and is back and Romeo:
I am such a, a common unto your means
And here the gates of't not fables and Forth
```

All networks besides STIRNN learned the structure of the Shakespear's plays quite well, STIRNN obviously looked at it from a different angle
and rather then merely imitating Shakespear came up with a few original structures on its own: `eie,ehsc so r ep:aln nr  ee cerfendAo`.

## Notes

* `graph.py` contains a small class `Grapher` that I use to update and display graph during the training.
* `util.py` contains few functions to process the input to batches and encode them using one-hot encoding

## License

MIT
