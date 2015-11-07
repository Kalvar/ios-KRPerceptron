## About

KRPerceptron implemented non-linear and linear algorithm in perceptron model of machine learning.

#### Podfile

```ruby
platform :ios, '7.0'
pod "KRPerceptron", "~> 1.0.1"
```

## How To Get Started & Samples

#### Import
``` objective-c
#import "KRPerceptron.h"
```

#### Identify Numbers
``` objective-c
KRPerceptron *_perceptron   = [KRPerceptron sharedPerceptron];
_perceptron.activeFunction  = KRPerceptronActiveFunctionBySigmoid;
_perceptron.maxIteration    = 1000;
_perceptron.learningRate    = 0.8f;
_perceptron.runOnMainThread = YES;

// Number 1
[_perceptron addPatterns:@[@0, @0, @0, @0, @0, @0, @0, @0, @0, @0, @0, @0,
                           @0, @0, @0, @0, @0, @0, @0, @0, @0, @0, @0, @0,
                           @0, @0, @0, @1, @1, @1, @1, @1, @1, @1, @1, @1]
                 outputs:@[@1, @0, @0, @0, @0, @0, @0, @0, @0, @0]];

// Number 2
[_perceptron addPatterns:@[@1, @0, @0, @0, @1, @1, @1, @1, @1, @1, @0, @0,
                           @0, @1, @0, @0, @0, @1, @1, @0, @0, @0, @1, @0,
                           @0, @0, @1, @1, @1, @1, @1, @1, @0, @0, @0, @1]
                 outputs:@[@0, @1, @0, @0, @0, @0, @0, @0, @0, @0]];

// Number 3
[_perceptron addPatterns:@[@1, @0, @0, @0, @1, @0, @0, @0, @1, @1, @0, @0,
                           @0, @1, @0, @0, @0, @1, @1, @0, @0, @0, @1, @0,
                           @0, @0, @1, @1, @1, @1, @1, @1, @1, @1, @1, @1]
                 outputs:@[@0, @0, @1, @0, @0, @0, @0, @0, @0, @0]];

// Number 4
[_perceptron addPatterns:@[@1, @1, @1, @1, @1, @0, @0, @0, @0, @0, @0, @0,
                           @0, @1, @0, @0, @0, @0, @0, @0, @0, @0, @1, @0,
                           @0, @0, @0, @1, @1, @1, @1, @1, @1, @1, @1, @1]
                 outputs:@[@0, @0, @0, @1, @0, @0, @0, @0, @0, @0]];

// Number 5
[_perceptron addPatterns:@[@1, @1, @1, @1, @1, @0, @0, @0, @1, @1, @0, @0,
                           @0, @1, @0, @0, @0, @1, @1, @0, @0, @0, @1, @0,
                           @0, @0, @1, @1, @0, @0, @0, @1, @1, @1, @1, @1]
                 outputs:@[@0, @0, @0, @0, @1, @0, @0, @0, @0, @0]];

// Number 6
[_perceptron addPatterns:@[@1, @1, @1, @1, @1, @1, @1, @1, @1, @1, @0, @0,
                           @0, @1, @0, @0, @0, @1, @1, @0, @0, @0, @1, @0,
                           @0, @0, @1, @1, @0, @0, @0, @1, @1, @1, @1, @1]
                 outputs:@[@0, @0, @0, @0, @0, @1, @0, @0, @0, @0]];

// Number 7
[_perceptron addPatterns:@[@1, @0, @0, @0, @0, @0, @0, @0, @0, @1, @0, @0,
                           @0, @0, @0, @0, @0, @0, @1, @0, @0, @0, @0, @0,
                           @0, @0, @0, @1, @1, @1, @1, @1, @1, @1, @1, @1]
                 outputs:@[@0, @0, @0, @0, @0, @0, @1, @0, @0, @0]];

// Number 8
[_perceptron addPatterns:@[@1, @1, @1, @1, @1, @1, @1, @1, @1, @1, @0, @0,
                           @0, @1, @0, @0, @0, @1, @1, @0, @0, @0, @1, @0,
                           @0, @0, @1, @1, @1, @1, @1, @1, @1, @1, @1, @1]
                 outputs:@[@0, @0, @0, @0, @0, @0, @0, @1, @0, @0]];

// Number 9
[_perceptron addPatterns:@[@1, @1, @1, @1, @1, @0, @0, @0, @0, @1, @0, @0,
                           @0, @1, @0, @0, @0, @0, @1, @0, @0, @0, @1, @0,
                           @0, @0, @0, @1, @1, @1, @1, @1, @1, @1, @1, @1]
                 outputs:@[@0, @0, @0, @0, @0, @0, @0, @0, @1, @0]];

// Number 0
[_perceptron addPatterns:@[@1, @1, @1, @1, @1, @1, @1, @1, @1, @1, @0, @0,
                           @0, @0, @0, @0, @0, @1, @1, @0, @0, @0, @0, @0,
                           @0, @0, @1, @1, @1, @1, @1, @1, @1, @1, @1, @1]
                 outputs:@[@0, @0, @0, @0, @0, @0, @0, @0, @0, @1]];

// Random settings
[_perceptron randomWeightsAndBiases];

__block typeof(_perceptron) _weakPerceptron = _perceptron;
[_perceptron setTrainingIteraion:^(NSInteger iteration, NSArray *weights) {
    NSLog(@"%li iteration, weights : %@", iteration, weights);
}];

[_perceptron trainingWithCompletion:^(BOOL success, NSArray *weights, NSInteger totalIteration) {
    if( success )
    {
        NSLog(@"total Iteration : %li", totalIteration);
        NSLog(@"Trained weights : %@", weights);
        
        // Verified number " 7 " which has some defects.
        NSArray *_verifyPatterns = @[@1, @1, @1, @0, @0, @0, @0, @0, @0, @1, @0, @0,
                                     @0, @0, @0, @0, @0, @0, @1, @0, @0, @0, @0, @0,
                                     @0, @0, @0, @1, @1, @1, @1, @1, @1, @1, @1, @1];
        [_weakPerceptron directOutputAtPatterns:_verifyPatterns
                                     completion:^(NSInteger iteration, NSArray *networkOutputs, NSArray *inputs) {
                                         NSLog(@"Verified networkOutputs : %@", networkOutputs);
                                     }];
    }
}];
```

## Version

V1.0.1

## LICENSE

MIT.

