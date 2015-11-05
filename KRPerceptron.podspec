Pod::Spec.new do |s|
  s.name         = "KRPerceptron"
  s.version      = "1.0.0"
  s.summary      = "Non-linear perceptron learning model of machine learning."
  s.description  = <<-DESC
                   KRPerceptron implemented non-linear and linear algorithm in perceptron of machine learning.
                   DESC
  s.homepage     = "https://github.com/Kalvar/ios-KRPerceptron"
  s.license      = { :type => 'MIT', :file => 'LICENSE' }
  s.author       = { "Kalvar Lin" => "ilovekalvar@gmail.com" }
  s.social_media_url = "https://twitter.com/ilovekalvar"
  s.source       = { :git => "https://github.com/Kalvar/ios-KRPerceptron.git", :tag => s.version.to_s }
  s.platform     = :ios, '7.0'
  s.requires_arc = true
  s.public_header_files = 'ML/*.h'
  s.source_files = 'ML/*.{h,m}'
  s.frameworks   = 'Foundation'
end 