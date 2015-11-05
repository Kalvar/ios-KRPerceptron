//
//  ViewController.m
//  KRPerceptron
//
//  Created by Kalvar Lin on 2015/11/3.
//  Copyright © 2015年 Kalvar Lin. All rights reserved.
//

#import "ViewController.h"
#import "KRPerceptron.h"

@interface ViewController ()

@end

@implementation ViewController

//To learn and verify numbers 0 to 9. And only setups patterns and output goals, and 10 outputs.
-(void)indentifyNumbers
{
    KRPerceptron *_perceptron  = [KRPerceptron sharedPerceptron];
    _perceptron.activeFunction = KRPerceptronActiveFunctionBySigmoid;
    _perceptron.maxIteration   = 1000;
    _perceptron.learningRate   = 0.8f;
    
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
        //NSLog(@"%li iteration, weights : %@", iteration, weights);
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
}

- (void)viewDidLoad {
    [super viewDidLoad];
    [self indentifyNumbers];
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
