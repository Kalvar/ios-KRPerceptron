//
//  KRPerceptron.h
//  KRPerceptron
//
//  Created by Kalvar Lin on 2015/11/3.
//  Copyright © 2015年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

typedef enum KRPerceptronActiveFunctions
{
    KRPerceptronActiveFunctionBySgn  = 0,
    KRPerceptronActiveFunctionByTanh,
    KRPerceptronActiveFunctionBySigmoid
}KRPerceptronActiveFunctions;

typedef void(^KRPerceptronCompletion)(BOOL success, NSArray *weights, NSInteger totalIteration);
typedef void(^KRPerceptronIteration)(NSInteger iteration, NSArray *weights);
typedef void(^KRPerceptronPattern)(NSInteger iteration, NSArray *networkOutputs, NSArray *inputs);

@interface KRPerceptron : NSObject

@property (nonatomic, strong) NSMutableArray *patterns;
@property (nonatomic, strong) NSMutableArray *targets;
@property (nonatomic, strong) NSMutableArray *weights;
@property (nonatomic, strong) NSMutableArray *biases;
@property (nonatomic, strong) NSArray *networkOutputs;
@property (nonatomic, assign) float learningRate;
@property (nonatomic, assign) NSInteger maxIteration;
@property (nonatomic, assign) NSInteger totalIteration;
@property (nonatomic, assign) float convergenceValue;
@property (nonatomic, assign) NSInteger countOutputNets;
@property (nonatomic, assign) BOOL runOnMainThread; // Suggets run on Main-Thread to train the network.

@property (nonatomic, assign) KRPerceptronActiveFunctions activeFunction;

@property (nonatomic, copy) KRPerceptronCompletion trainingCompletion;
@property (nonatomic, copy) KRPerceptronIteration trainingIteraion;
@property (nonatomic, copy) KRPerceptronPattern trainingPattern;

+(instancetype)sharedPerceptron;
-(instancetype)init;

-(void)addPatterns:(NSArray *)_inputs outputs:(NSArray *)_outputs;
-(void)addWeights:(NSArray *)_lineWeights;
-(void)addBias:(double)_netBiase;
-(void)addBiases:(NSArray *)_netBiases;

-(void)randomWeightsAndBiases;

-(void)training;
-(void)trainingWithCompletion:(KRPerceptronCompletion)_completion;
-(void)directOutputAtPatterns:(NSArray *)_inputs completion:(KRPerceptronPattern)_completion;

-(void)reset;

-(void)setTrainingCompletion:(KRPerceptronCompletion)_block;
-(void)setTrainingIteraion:(KRPerceptronIteration)_block;
-(void)setTrainingPattern:(KRPerceptronPattern)_block;

@end
