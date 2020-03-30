# Logic Gate Simulator with ANN
Lately I've been fascinated with the idea of neural architectures and the way it enables us to model human brain into computer and train it on complicated tasks that would otherwise be out of reach for traditional programming paradigms. Considering the simulation of logic gates, it is an easy task for traditional programming which can be implemeneted using if else selection statements, but I wanted to build it using artificial neural network. This will simulate how we as humans learn logic. 

Table of Contents
-------------
1. Understanding Neural Architecture
2. Designing the Training Dataset
3. Selection of Weights
4. Barebones empty neural network
5. Implementing forward propagation
6. Backpropagation and learning data
7. Saving the learned weights
8. Testing
9. Analysis of accuracy
                

Understanding the Neural Architecture
-------------
First question arises as to how does neural architecture work. Well, it's best to draw parallels of it from a new born child. How do they learn? They learn by observing their surroundings, for exampe, their parents, or sibling if they have any. Watching with a curious eye as to how do they walk, talk, or do, child's mind starts to adapt its actions to mimic those of the others. 

![](https://github.com/harisrab/logicSim_ANN/blob/master/examples/giphy.gif)
> Children are the best immitators in the world.

If we are negligent of the vast underlying complexity of human mind, it becomes easy for us to say that we can replicate human mind in machine code but that would be a far fetched fantasy. 
It can be very motivating to think of such fantasies and form goals that seem unachievable. Science has sought for us ways for achieving these goals by distilling them down into fundamental truths and reason up from there. This is exactly the right way of thinking when it comes to building something close in complexity to neural network. It makes perfect sense to start from fundamental decision makers in the world of compter science, logic gates, and simulate them in a way that is different to the way how it is traditionally done.

https://www.youtube.com/watch?v=aircAruvnKk

The link above will guide you through the basics of how neural networks work and it has been an inspiration for me how he approaches the fundamental truths. Now we will design our inputs and outputs to create a dataset upon which our neural network can be trained to simulate and give correct predictions.

Designing the Training Dataset
-------------
![](https://github.com/harisrab/logicSim_ANN/blob/master/examples/globe.gif)

In the real world, for a child, data is observed, collected, and computed in the most seamless fashion, but here when we are to design something as simple as logic gates, we need to look for ways it would enable us to understand the data and be perfect so that our neural network does not have problems processing it.

When in our exams, we are asked to draw the truth table for logic gate, what information do we seek:

1. Type of logic gate
2. Input states


Since here we are dealing with two input logic gates, our neural network needs to identify six different types of them and then take the inputs to tell us what they compute. We can use 3 bit code to identify a logic gate and 2 successive bits to identify inputs. Therefore our neural network stands at 5 inputs and one output for the answer. Well as far as the middle layer is concerned we use it to extract more detailed features and establish a correleation between the input and output if there exist none. Below is the picture of neural network we are going to design. 

![](https://github.com/harisrab/logicSim_ANN/blob/master/examples/neural_net.jpg)
> Shape of the neural network

**Dataset:**


|Input   | Output |   | Input | Output||     Input | Output   |
|--------|--------|---|------ |:------:|---|-------|:--------:|
| 00000  |  0     |   | 01000 | 0      |   | 10000 | 0        |
| 00001  |  1     |   | 01001 | 0      |   | 10001 | 1        |
| 00010  |  1     |   | 01010 | 0      |   | 10010 | 1        |
| 00011  |  1     |   | 01011 | 1      |   | 10011 | 0        |
| 00100  |  1     |   | 01100 | 1      |   | 10100 | 1        |
| 00101  |  0     |   | 01100 | 1      |   | 10101 | 0        |
| 00110  |  0     |   | 01100 | 1      |   | 10110 | 0        |
| 00111  |  0     |   | 01100 | 0      |   | 10111 | 1        |

First three bits correspond to each different gate. Here is a table that gives that information:

|Input   | Gate   |  
|--------|--------|
| 000    |  OR    |
| 001    |  NOR   |
| 010    |  AND   |
| 011    |  NAND  |
| 100    |  XOR   |
| 101    |  NXOR  |

#H1 header
####Ordered list
###H3 header
####H4 header
#####H5 header
######H6 header
#Heading 1 link [Heading link](https://github.com/pandao/editor.md "Heading link")
##Heading 2 link [Heading link](https://github.com/pandao/editor.md "Heading link")
###Heading 3 link [Heading link](https://github.com/pandao/editor.md "Heading link")
####Heading 4 link [Heading link](https://github.com/pandao/editor.md "Heading link") Heading link [Heading link](https://github.com/pandao/editor.md "Heading link")
#####Heading 5 link [Heading link](https://github.com/pandao/editor.md "Heading link")
######Heading 6 link [Heading link](https://github.com/pandao/editor.md "Heading link")




##Headers (Underline)

H1 Header (Underline)
=============

H2 Header (Underline)
-------------

###Characters
                
----

~~Strikethrough~~ <s>Strikethrough (when enable html tag decode.)</s>
*Italic*      _Italic_
**Emphasis**  __Emphasis__
***Emphasis Italic*** ___Emphasis Italic___

Superscript: X<sub>2</sub>，Subscript: O<sup>2</sup>

**Abbreviation(link HTML abbr tag)**

The <abbr title="Hyper Text Markup Language">HTML</abbr> specification is maintained by the <abbr title="World Wide Web Consortium">W3C</abbr>.

###Blockquotes

> Blockquotes

Paragraphs and Line Breaks
                    
> "Blockquotes Blockquotes", [Link](http://localhost/)。

###Links

[Links](http://localhost/)

[Links with title](http://localhost/ "link title")

`<link>` : <https://github.com>

[Reference link][id/name] 

[id/name]: http://link-url/

GFM a-tail link @pandao

###Code Blocks (multi-language) & highlighting

####Inline code

`$ npm install marked`

####Code Blocks (Indented style)

Indented 4 spaces, like `<pre>` (Preformatted Text).

    <?php
        echo "Hello world!";
    ?>
    
Code Blocks (Preformatted text):

    | First Header  | Second Header |
    | ------------- | ------------- |
    | Content Cell  | Content Cell  |
    | Content Cell  | Content Cell  |

####Javascript　

```javascript
function test(){
	console.log("Hello world!");
}
 
(function(){
    var box = function(){
        return box.fn.init();
    };

    box.prototype = box.fn = {
        init : function(){
            console.log('box.init()');

			return this;
        },

		add : function(str){
			alert("add", str);

			return this;
		},

		remove : function(str){
			alert("remove", str);

			return this;
		}
    };
    
    box.fn.init.prototype = box.fn;
    
    window.box =box;
})();

var testBox = box();
testBox.add("jQuery").remove("jQuery");
```

####HTML code

```html
<!DOCTYPE html>
<html>
    <head>
        <mate charest="utf-8" />
        <title>Hello world!</title>
    </head>
    <body>
        <h1>Hello world!</h1>
    </body>
</html>
```

###Images

Image:

![](https://pandao.github.io/editor.md/examples/images/4.jpg)

> Follow your heart.

![](https://pandao.github.io/editor.md/examples/images/8.jpg)

> 图为：厦门白城沙滩 Xiamen

图片加链接 (Image + Link)：

[![](https://pandao.github.io/editor.md/examples/images/7.jpg)](https://pandao.github.io/editor.md/examples/images/7.jpg "李健首张专辑《似水流年》封面")

> 图为：李健首张专辑《似水流年》封面
                
----

###Lists

####Unordered list (-)

- Item A
- Item B
- Item C
     
####Unordered list (*)

* Item A
* Item B
* Item C

####Unordered list (plus sign and nested)
                
+ Item A
+ Item B
    + Item B 1
    + Item B 2
    + Item B 3
+ Item C
    * Item C 1
    * Item C 2
    * Item C 3


                    
###Tables
                    
First Header  | Second Header
------------- | -------------
Content Cell  | Content Cell
Content Cell  | Content Cell 



| Function name | Description                    |
| ------------- | ------------------------------ |
| `help()`      | Display the help window.       |
| `destroy()`   | **Destroy your computer!**     |

| Item      | Value |
| --------- | -----:|
| Computer  | $1600 |
| Phone     |   $12 |
| Pipe      |    $1 |

| Left-Aligned  | Center Aligned  | Right Aligned |
| :------------ |:---------------:| -----:|
| col 3 is      | some wordy text | $1600 |
| col 2 is      | centered        |   $12 |
| zebra stripes | are neat        |    $1 |
                
----

####HTML entities

&copy; &  &uml; &trade; &iexcl; &pound;
&amp; &lt; &gt; &yen; &euro; &reg; &plusmn; &para; &sect; &brvbar; &macr; &laquo; &middot; 

X&sup2; Y&sup3; &frac34; &frac14;  &times;  &divide;   &raquo;

18&ordm;C  &quot;  &apos;

##Escaping for Special Characters

\*literal asterisks\*

##Markdown extras

###GFM task list

- [x] GFM task list 1
- [x] GFM task list 2
- [ ] GFM task list 3
    - [ ] GFM task list 3-1
    - [ ] GFM task list 3-2
    - [ ] GFM task list 3-3
- [ ] GFM task list 4
    - [ ] GFM task list 4-1
    - [ ] GFM task list 4-2

###Emoji mixed :smiley:

> Blockquotes :star:

####GFM task lists & Emoji & fontAwesome icon emoji & editormd logo emoji :editormd-logo-5x:

- [x] :smiley: @mentions, :smiley: #refs, [links](), **formatting**, and <del>tags</del> supported :editormd-logo:;
- [x] list syntax required (any unordered or ordered list supported) :editormd-logo-3x:;
- [x] [ ] :smiley: this is a complete item :smiley:;
- [ ] []this is an incomplete item [test link](#) :fa-star: @pandao; 
- [ ] [ ]this is an incomplete item :fa-star: :fa-gear:;
    - [ ] :smiley: this is an incomplete item [test link](#) :fa-star: :fa-gear:;
    - [ ] :smiley: this is  :fa-star: :fa-gear: an incomplete item [test link](#);
            
###TeX(LaTeX)
   
$$E=mc^2$$

Inline $$E=mc^2$$ Inline，Inline $$E=mc^2$$ Inline。

$$\(\sqrt{3x-1}+(1+x)^2\)$$
                    
$$\sin(\alpha)^{\theta}=\sum_{i=0}^{n}(x^i + \cos(f))$$
                
###FlowChart

```flow
st=>start: Login
op=>operation: Login operation
cond=>condition: Successful Yes or No?
e=>end: To admin

st->op->cond
cond(yes)->e
cond(no)->op
```

###Sequence Diagram
                    
```seq
Andrew->China: Says Hello 
Note right of China: China thinks\nabout it 
China-->Andrew: How are you? 
Andrew->>China: I am good thanks!
```

###End
