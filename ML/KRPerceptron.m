//
//  KRPerceptron.m
//  KRPerceptron
//
//  Created by Kalvar Lin on 2015/11/3.
//  Copyright © 2015年 Kalvar Lin. All rights reserved.
//

#import "KRPerceptron.h"

#define DEFAULT_LEARNING_RATE     0.5f
#define DEFAULT_MAX_ITERATION     1
#define DEFAULT_CONVERGENCE_VALUE 0.001f
#define DEFAULT_ITERATION         0
#define DEFAULT_RANDOM_MAX        0.5f
#define DEFAULT_RANDOM_MIN        -0.5f

static NSInteger kPatternOutputResults = 0;
static NSInteger kPatternOutputErrors  = 1;

@interface KRPerceptron ()

@property (nonatomic, assign) NSInteger iteration;
@property (nonatomic, strong) NSMutableArray *iterationErrors; // 所有迭代的 Output Error Values

@end

@implementation KRPerceptron (fixCostFunctions)

-(double)_mse:(NSArray *)_iterationErrors
{
    double _sumError = 0.0f;
    for( NSArray *_patternErrors in _iterationErrors )
    {
        for( NSNumber *_outputError in _patternErrors )
        {
            _sumError += ( [_outputError doubleValue] * [_outputError doubleValue] );
        }
    }
    return _sumError;
}

@end

@implementation KRPerceptron (fixMaths)

-(double)_randomMax:(double)_maxValue min:(double)_minValue
{
    return ((double)arc4random() / ( RAND_MAX * 2.0f ) ) * (_maxValue - _minValue) + _minValue;
}

@end

@implementation KRPerceptron (fixActiviations)

// Tanh() which scope is [-1.0, 1.0]
-(double)_fOfTanh:(double)_x
{
    return ( 2.0f / ( 1.0f + pow(M_E, (-2.0f * _x)) ) ) - 1.0f; // -2.0f = 入
}

// Sigmoid() whici scope is [0.0, 1.0]
-(double)_fOfSigmoid:(double)_x
{
    return ( 1.0f / ( 1.0f + pow(M_E, (-1.0f * _x)) ) ); // -1.0f = 入
}

// SGN() which scope is (-1, 1) or (0, 1)
-(double)_fOfSgn:(double)_sgnValue
{
    return ( _sgnValue >= 0.0f ) ? 1.0f : -1.0f;
}

-(double)_activateOutput:(double)_outputValue
{
    double _activatedValue = 0.0f;
    switch (self.activeFunction)
    {
        case KRPerceptronActiveFunctionBySigmoid:
            _activatedValue = [self _fOfSigmoid:_outputValue];
            break;
        case KRPerceptronActiveFunctionByTanh:
            _activatedValue = [self _fOfTanh:_outputValue];
            break;
        default:
            _activatedValue = [self _fOfSgn:_outputValue];
            break;
    }
    return _activatedValue;
}

// f'(net), 偏微分
-(double)_fDashOfNetOutput:(double)_netOutput
{
    double _dashOfNet = 0.0f;
    switch (self.activeFunction)
    {
        case KRPerceptronActiveFunctionBySigmoid:
            _dashOfNet = _netOutput * ( 1 - _netOutput );
            break;
        case KRPerceptronActiveFunctionByTanh:
            _dashOfNet = 1 - ( _netOutput * _netOutput );
            //_dashOfNet /= (入 / 2)
            break;
        default:
            _dashOfNet = _netOutput;
            break;
    }
    return _dashOfNet;
}

@end

@implementation KRPerceptron (fixPerceptron)

// Return [0] = output-net outouts, [1] = output-net output errors
-(NSArray *)_fOfNetWithInputs:(NSArray *)_inputs patternIndex:(NSInteger)_patternIndex
{
    NSMutableArray *_outputs  = [NSMutableArray new];
    NSMutableArray *_errors   = [NSMutableArray new];
    NSArray *_patternTargets  = [self.targets objectAtIndex:_patternIndex];
    // One output, one net
    NSInteger _outputIndex    = 0;
    for( NSNumber *_outputTarget in _patternTargets )
    {
        double _netBias       = [[self.biases objectAtIndex:_outputIndex] doubleValue];
        double _targetValue   = [_outputTarget doubleValue];
        double _sum           = 0.0f;
        NSInteger _inputIndex = 0;
        for( NSNumber *_inputValue in _inputs )
        {
            // 取出第幾個 Input Value 的權重 Array -> 之後再取出對應第幾個 Output Net 的權重值
            NSArray *_inputWeights  = [self.weights objectAtIndex:_inputIndex];
            NSNumber *_lineWeight   = [_inputWeights objectAtIndex:_outputIndex];
            _sum                   += ( [_inputValue doubleValue] * [_lineWeight doubleValue] );
            ++_inputIndex;
        }
        // Plus net bias
        _sum                   += _netBias;
        // Use active function to activate the output-value of net
        double _activatedOutput = [self _activateOutput:_sum];
        // Calculate the error value
        double _errorValue      = _targetValue - _activatedOutput;
        // Adding output and error
        [_outputs addObject:[NSNumber numberWithDouble:_activatedOutput]];
        [_errors addObject:[NSNumber numberWithDouble:_errorValue]];
        ++_outputIndex;
    }
    return @[_outputs, _errors];
}

-(void)_turningWeightsByInputs:(NSArray *)_inputs patternOutputs:(NSArray *)_patternOutputs
{
    NSArray *_weights        = [self.weights copy];
    float _learningRate      = self.learningRate;
    NSArray *_netOutputs     = [_patternOutputs objectAtIndex:kPatternOutputResults];
    NSArray *_outputErrors   = [_patternOutputs objectAtIndex:kPatternOutputErrors];
    // 依照每組 Inputs 的權重更新回去
    NSInteger _inputIndex    = 0;
    for( NSArray *_inputWeights in _weights )
    {
        double _inputValue              = [[_inputs objectAtIndex:_inputIndex] doubleValue];
        NSMutableArray *_refreshWeights = [NSMutableArray new];
        NSInteger _outputIndex          = 0;
        // Refreshing xj owns line-weights
        for( NSNumber *_lineWeight in _inputWeights )
        {
            double _outputValue = [[_netOutputs objectAtIndex:_outputIndex] doubleValue];
            double _errorValue  = [[_outputErrors objectAtIndex:_outputIndex] doubleValue];
            // Formula : -(learning rate) * -(netj output error) * f'(netj) * xj (前頭負負得正)
            double _deltaWeight = _learningRate * _errorValue * [self _fDashOfNetOutput:_outputValue] * _inputValue;
            double _newWeight   = [_lineWeight doubleValue] + _deltaWeight;
            [_refreshWeights addObject:[NSNumber numberWithDouble:_newWeight]];
            ++_outputIndex;
        }
        [self.weights replaceObjectAtIndex:_inputIndex withObject:_refreshWeights];
        ++_inputIndex;
    }
    _weights = nil;
    [self.iterationErrors addObject:_outputErrors];
}

-(void)_loopingPatterns:(NSArray *)_patterns tuningWeights:(BOOL)_tuningWeights networkOutputHandler:(KRPerceptronPattern)_outputHandler
{
    //self.trainingPattern = _outputHandler;
    NSInteger _index     = 0;
    for( NSArray *_inputs in _patterns )
    {
        NSArray *_patternOutputs = [self _fOfNetWithInputs:_inputs patternIndex:_index];
        self.networkOutputs      = [_patternOutputs objectAtIndex:kPatternOutputResults];
        if( nil != _outputHandler )
        {
            _outputHandler(self.totalIteration, self.networkOutputs, _inputs);
        }
        // Wants to tune weights
        if( _tuningWeights )
        {
            [self _turningWeightsByInputs:_inputs patternOutputs:_patternOutputs];
        }
        ++_index;
    }
}

-(void)_doTraining
{
    ++self.iteration;
    [self.iterationErrors removeAllObjects];
    [self _loopingPatterns:self.patterns tuningWeights:YES networkOutputHandler:nil];
    if( self.iteration >= self.maxIteration || [self _mse:self.iterationErrors] <= self.convergenceValue )
    {
        if( nil != self.trainingCompletion )
        {
            self.trainingCompletion(YES, self.weights, self.iteration);
        }
    }
    else
    {
        if( nil != self.trainingIteraion )
        {
            self.trainingIteraion(self.iteration, self.weights);
        }
        [self _doTraining];
    }
}

@end


@implementation KRPerceptron

+(instancetype)sharedPerceptron
{
    static dispatch_once_t pred;
    static KRPerceptron *_object = nil;
    dispatch_once(&pred, ^{
        _object = [[KRPerceptron alloc] init];
    });
    return _object;
}

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        _learningRate       = DEFAULT_LEARNING_RATE;
        _weights            = [NSMutableArray new];
        _patterns           = [NSMutableArray new];
        _targets            = [NSMutableArray new];
        _networkOutputs     = [NSMutableArray new];
        
        _maxIteration       = DEFAULT_MAX_ITERATION;
        _totalIteration     = DEFAULT_ITERATION;
        _convergenceValue   = DEFAULT_CONVERGENCE_VALUE;
        _biases             = [NSMutableArray new];
        _networkOutputs     = nil;
        _activeFunction     = KRPerceptronActiveFunctionBySigmoid;
        
        _runOnMainThread    = YES; // 是否要跑在 MainThread 裡做訓練
        
        _iteration          = DEFAULT_ITERATION;
        _iterationErrors    = [NSMutableArray new];
        
        _trainingCompletion = nil;
        _trainingIteraion   = nil;
        _trainingPattern    = nil;
    }
    return self;
}

#pragma --mark Public Methods
-(void)addPatterns:(NSArray *)_inputs outputs:(NSArray *)_outputs
{
    [_patterns addObject:_inputs];
    [_targets addObject:_outputs];
}

-(void)addWeights:(NSArray *)_lineWeights
{
    [_weights addObject:_lineWeights]; // @[@[w14, w15], @[w24, w25] ...]
}

-(void)addBias:(double)_netBiase
{
    [_biases addObject:[NSNumber numberWithDouble:_netBiase]];
}

// 也代表 Output Nets 個數，須跟 Patterns 的 Outputs (Targets) 數量相等
-(void)addBiases:(NSArray *)_netBiases
{
    [_biases addObjectsFromArray:_netBiases]; // @[ net1 bias, net2 bias, ...]
}

-(void)randomWeightsAndBiases
{
    // 先清空歸零
    [_weights removeAllObjects];
    [_biases removeAllObjects];
    
    // 有幾個輸出值，就有幾顆運作處理的神經元
    NSInteger _outputCount = [[_targets firstObject] count];
    
    // 輸入層權重初始化規則 : ( 0.5 / 此層神經元個數 ) ~ ( -0.5 / 此層神經元個數 )
    NSInteger _inputNetCount = [[_patterns firstObject] count];
    double _inputMax         = DEFAULT_RANDOM_MAX / _inputNetCount;
    double _inputMin         = DEFAULT_RANDOM_MIN / _inputNetCount;
    for( int i=0; i<_inputNetCount; i++ )
    {
        NSMutableArray *_randomWeights = [NSMutableArray new];
        for( int j=0; j<_outputCount; j++ )
        {
            [_randomWeights addObject:[NSNumber numberWithDouble:[self _randomMax:_inputMax min:_inputMin]]];
        }
        [_weights addObject:_randomWeights];
    }
    
    // 輸出層神經元的偏權值
    for( int _i=0; _i<_outputCount; _i++ )
    {
        [_biases addObject:[NSNumber numberWithDouble:[self _randomMax:_inputMax min:_inputMin]]];
    }
}

-(void)training
{
    // If you wanna run on Main-Thread that training iterations could be over 2,000.
    // 跑在 Main-Thread 較能保證其運行不崩潰，也能做超過 2,000 迭代的連續運算
    if( _runOnMainThread )
    {
        dispatch_async(dispatch_get_main_queue(), ^{
            [self _doTraining];
        });
    }
    else
    {
        // 反之，如果只是想要做極短暫的訓練 (100 迭代以下)，那跑在 Async-Thread 裡是穩定可行的
        dispatch_queue_t queue = dispatch_queue_create("com.krperceptron.train", NULL);
        dispatch_async(queue, ^(void){
            [self _doTraining];
        });
    }
}

-(void)trainingWithCompletion:(KRPerceptronCompletion)_completion
{
    _trainingCompletion = _completion;
    [self training];
}

-(void)directOutputAtPatterns:(NSArray *)_inputs completion:(KRPerceptronPattern)_completion
{
    _trainingCompletion = nil;
    _iteration          = 1;
    [self _loopingPatterns:@[_inputs] tuningWeights:NO networkOutputHandler:_completion];
}

-(void)reset
{
    [_patterns removeAllObjects];
    [_targets removeAllObjects];
    [_weights removeAllObjects];
    [_biases removeAllObjects];
    _networkOutputs   = nil;
    _maxIteration     = DEFAULT_MAX_ITERATION;
    _iteration        = DEFAULT_ITERATION;
    _convergenceValue = DEFAULT_CONVERGENCE_VALUE;
    _learningRate     = DEFAULT_LEARNING_RATE;
}

#pragma --mark Block Setters
-(void)setTrainingCompletion:(KRPerceptronCompletion)_block
{
    _trainingCompletion = _block;
}

-(void)setTrainingIteraion:(KRPerceptronIteration)_block
{
    _trainingIteraion = _block;
}

-(void)setTrainingPattern:(KRPerceptronPattern)_block
{
    _trainingPattern = _block;
}

#pragma --mark Getters
-(NSInteger)countOutputNets
{
    return [_biases count];
}

-(NSInteger)totalIteration
{
    return _iteration;
}

@end
