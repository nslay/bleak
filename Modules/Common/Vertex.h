/*-
 * Copyright (c) 2017 Nathan Lay (enslay@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#ifndef BLEAK_VERTEX_H
#define BLEAK_VERTEX_H

#include <iostream>
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <functional>
#include <unordered_map>
#include "Common.h"
#include "Property.h"
#include "Edge.h"
#include "Database.h"

// Only needed if your class inherits from a templated class with unresolved template parameters
#define bleakForwardVertexTypedefs() \
  typedef typename SuperType::VertexType VertexType; \
  typedef typename SuperType::EdgeType EdgeType; \
  typedef typename SuperType::ArrayType ArrayType; \
  typedef typename SuperType::OutputMapType OutputMapType; \
  typedef typename SuperType::InputMapType InputMapType; \
  typedef typename SuperType::PropertyMapType PropertyMapType

#define bleakNewAbstractVertex(thisClass, superClass, ...) \
  protected: \
  virtual void OnCreateWithNew() override { \
    superClass :: OnCreateWithNew(); \
    int forNewVertexMacro_ = 0; \
    __VA_ARGS__ ; \
  } \
  public: \
    typedef superClass SuperType; \
    typedef thisClass SelfType; \
    using SuperType::shared_from_this; \
    using SuperType::GetOutput; \
    using SuperType::SetInput; \
    using SuperType::GetInput; \
    using SuperType::HasInput; \
    using SuperType::HasOutput; \
    using SuperType::GetAllInputs; \
    using SuperType::GetAllOutputs; \
    using SuperType::GetProperty; \
    using SuperType::SetProperty; \
    using SuperType::GetName; \
    using SuperType::IsLeaf

#define bleakNewVertex(thisClass, superClass, ...) \
  static constexpr const char * GetTypeName() { return #thisClass ; } \
  static std::shared_ptr< thisClass > New() { \
    std::shared_ptr< thisClass > p_clVertex(new ( thisClass ) ()); \
    p_clVertex->OnCreateWithNew(); \
    return p_clVertex; \
  } \
  bleakNewAbstractVertex(thisClass, superClass, __VA_ARGS__)

#define bleakAddInput(name) forNewVertexMacro_; this->RegisterInput(name); forNewVertexMacro_
#define bleakAddOutput(name) forNewVertexMacro_; this->RegisterOutput(name); forNewVertexMacro_
#define bleakAddProperty(name, value) forNewVertexMacro_; this->RegisterProperty(name, value); forNewVertexMacro_
#define bleakAddGetterSetter(name, memberGetter, memberSetter) \
  forNewVertexMacro_; \
  this->RegisterGetterSetter(name, \
  std::bind( memberGetter , this, std::placeholders::_1), \
  std::bind( memberSetter , this, std::placeholders::_1) ); \
  forNewVertexMacro_

#define bleakGetAndCheckOutput(varName, outputName, ...) \
  std::shared_ptr<EdgeType> varName = this->GetOutput( outputName ); \
  if ( varName == nullptr ) { \
    std::cerr << this->GetName() << ": Error: Could not get output '" << outputName << "'." << std::endl; \
    return __VA_ARGS__ ; \
  }

#define bleakGetAndCheckInput(varName, inputName, ...) \
  std::shared_ptr<EdgeType> varName = this->GetInput( inputName ); \
  if( varName == nullptr ) { \
      std::cerr << this->GetName() << ": Error: Could not get input '" << inputName << "'." << std::endl; \
      return __VA_ARGS__ ; \
  }

namespace bleak {

template<typename RealType>
class Vertex : public std::enable_shared_from_this<Vertex<RealType>> {
public:
  typedef std::enable_shared_from_this<Vertex<RealType>> SuperType;
  typedef Vertex<RealType> VertexType;
  typedef Edge<RealType> EdgeType;
  typedef Array<RealType> ArrayType;
  typedef std::unordered_map<std::string, std::shared_ptr<EdgeType>> OutputMapType;
  typedef std::unordered_map<std::string, std::weak_ptr<EdgeType>> InputMapType;
  typedef std::unordered_map<std::string, std::unique_ptr<Property>> PropertyMapType;

  using SuperType::shared_from_this;

  virtual ~Vertex() {
    // Formally disconnect our outputs from targets' inputs (if any)
    for (auto &clPair : m_mOutputs) {
      const auto vTargets = clPair.second->GetAllTargets(); // This can change on UnsetInput(), so it must be a copy

      for (const auto &p_clWeakTarget : vTargets) {
        std::shared_ptr<VertexType> p_clTarget = p_clWeakTarget.first.lock();

        if (p_clTarget != nullptr) {
          std::string strInputAtTarget;

          // Could be multiple inputs with this edge
          while ((strInputAtTarget = p_clTarget->FindInputName(clPair.second)).size() > 0) {
            p_clTarget->UnsetInput(strInputAtTarget);
          }
        }
      }
    }

    // We can't safely call shared_from_this() in a destructor.
    // Thus we can't use UnsetInput() here! So let's just try to clean up expired targets in edges!
    for (auto &clPair : m_mInputs) {
      std::shared_ptr<EdgeType> p_clEdge = clPair.second.lock();

      if (p_clEdge != nullptr) {
        p_clEdge->RemoveExpiredTargets();
        clPair.second.reset();
      }
    }
  }

  virtual bool SetSizes() = 0;
  virtual bool Initialize() = 0;

  virtual void Forward() {
    if (m_bUseGPU)
      ForwardGPU();
    else
      ForwardCPU();
  }

  virtual void Backward() {
    if (m_bUseGPU)
      BackwardGPU();
    else
      BackwardCPU();
  }

  virtual void ForwardCPU() { }
  virtual void ForwardGPU() { ForwardCPU(); }

  virtual void BackwardCPU() { }
  virtual void BackwardGPU() { BackwardCPU(); }

  // This is a convenience function. Memory for vertices can be setup externally!
  bool Allocate(bool bForTraining) {
    for (auto &clPair : m_mOutputs) {
      if (!clPair.second->GetData().Allocate())
        return false;

      if (bForTraining && clPair.second->GetGradient().GetSize().Valid() && !clPair.second->GetGradient().Allocate())
        return false;
    }

    return true;
  }

  // These should only indicate a failure if a Database function legitimately fails or the data is unexpected!
  virtual bool SaveToDatabase(const std::unique_ptr<Transaction> &p_clTransaction) const {
    if(p_clTransaction == nullptr)
      return false;

    if (!m_bSaveOutputs)
      return true; // Nothing to do

    std::vector<double> vBuffer; // Save to doubles no matter what

    for (const auto &clOutputPair : m_mOutputs) {
      const std::string &strOutputName = clOutputPair.first;
      const auto &p_clEdge = clOutputPair.second;

      const ArrayType &clData = p_clEdge->GetData();

      if (!clData.Valid())
        continue;

      vBuffer.resize(clData.GetSize().Product());

      const std::string strKey = MakeDatabaseKey(strOutputName);

      std::cout << GetName() << ": Info: Saving output '" << strKey << "'." << std::endl;

      std::transform(clData.begin(), clData.end(), vBuffer.begin(), 
        [](const RealType &x) -> double {
          return (double)x;
        });

      p_clTransaction->Put(strKey, (uint8_t *)vBuffer.data(), sizeof(double)*vBuffer.size());
    }

    return p_clTransaction->Commit();
  }

  // Should be called after SetSize() and memory allocation at least!
  virtual bool LoadFromDatabase(const std::unique_ptr<Cursor> &p_clCursor) {
    if (p_clCursor == nullptr)
      return false;

    for (auto &clOutputPair : m_mOutputs) {
      const std::string &strOutputName = clOutputPair.first;
      const auto &p_clEdge = clOutputPair.second;

      ArrayType &clData = p_clEdge->GetData();

      if (!clData.Valid())
        continue;

      const std::string strKey = MakeDatabaseKey(strOutputName);

      if (!p_clCursor->Find(strKey))
        continue;

      std::cout << GetName() << ": Info: Loading output '" << strKey << "'." << std::endl;

      size_t bufferSize = 0;
      const double * const p_dBuffer = (double *)p_clCursor->Value(bufferSize);

      if (bufferSize != clData.GetSize().Product()*sizeof(double)) {
        std::cerr << GetName() << ": Error: Expected size " << clData.GetSize() << " (" << clData.GetSize().Product() * sizeof(double) << ") but got " << bufferSize << '.' << std::endl;
        return false;
      }

      std::transform(p_dBuffer, p_dBuffer + clData.GetSize().Product(), clData.begin(),
        [](const double &x) -> RealType {
          return RealType(x);
        });
    }

    return true;
  }

  void SetName(const std::string &strName) {
    m_strName = strName;
  }

  const std::string & GetName() const {
    return m_strName;
  }

  bool SetInput(const std::string &strName, const std::shared_ptr<EdgeType> &p_clEdge) {
    auto itr = m_mInputs.find(strName);

    if (itr == m_mInputs.end())
      return false;

    std::shared_ptr<EdgeType> p_clOldEdge = itr->second.lock();

    if (p_clEdge == p_clOldEdge)
      return true;

    if (p_clOldEdge != nullptr) {
      p_clOldEdge->RemoveTarget(shared_from_this());
      itr->second.reset();
    }

    if (p_clEdge != nullptr) {
      itr->second = p_clEdge;
      p_clEdge->AddTarget(shared_from_this());
    }

    return true;
  }

  void UnsetInput(const std::string &strName) {
    auto itr = m_mInputs.find(strName);

    if(itr == m_mInputs.end())
      return;

    std::shared_ptr<EdgeType> p_clEdge = itr->second.lock();

    if (p_clEdge != nullptr) {
      p_clEdge->RemoveTarget(shared_from_this());
      itr->second.reset();
    }
  }

  std::shared_ptr<EdgeType> GetOutput(const std::string &strName) const {
    auto itr = m_mOutputs.find(strName);
    return itr != m_mOutputs.end() ? itr->second : std::shared_ptr<EdgeType>();
  }

  std::shared_ptr<EdgeType> GetInput(const std::string &strName) const {
    auto itr = m_mInputs.find(strName);
    return itr != m_mInputs.end() ? itr->second.lock() : std::shared_ptr<EdgeType>();
  }

  const OutputMapType & GetAllOutputs() const {
    return m_mOutputs;
  }

  const InputMapType & GetAllInputs() const {
    return m_mInputs;
  }

  std::string FindInputName(const std::shared_ptr<EdgeType> &p_clEdge) const {
    if (p_clEdge == nullptr)
      return std::string();

    for (const auto &clPair : m_mInputs) {
      if (clPair.second.lock() == p_clEdge)
        return clPair.first;
    }

    return std::string();
  }

  std::string FindOutputName(const std::shared_ptr<EdgeType> &p_clEdge) const {
    if (p_clEdge == nullptr)
      return std::string();

    for (const auto &clPair : m_mOutputs) {
      if (clPair.second == p_clEdge)
        return clPair.first;
    }

    return std::string();
  }

  virtual bool HasOutput(const std::string &strName) const {
    return m_mOutputs.find(strName) != m_mOutputs.end();
  }

  virtual bool HasInput(const std::string &strName) const {
    return m_mInputs.find(strName) != m_mInputs.end();
  }

  virtual bool HasOutputs() const {
    return m_mOutputs.size() > 0;
  }

  virtual bool HasInputs() const {
    return m_mInputs.size() > 0;
  }

  template<typename ValueType>
  bool SetProperty(const std::string &strKey, const ValueType &value) {
    auto itr = m_mProperties.find(strKey);
    return itr != m_mProperties.end() ? itr->second->SetValue(value) : false;
  }

  template<typename ValueType>
  bool GetProperty(const std::string &strKey, ValueType &value) const {
    auto itr = m_mProperties.find(strKey);
    return itr != m_mProperties.end() ? itr->second->GetValue(value) : false;
  }

  const PropertyMapType & GetAllProperties() {
    return m_mProperties;
  }

  bool IsRoot() const {
    if(m_mInputs.empty())
      return true;

    for(const auto &clPair : m_mInputs) {
      std::shared_ptr<EdgeType> p_clEdge = clPair.second.lock();

      if(p_clEdge != nullptr && p_clEdge->GetSource() != nullptr)
        return false;
    }

    return true;
  }

  bool IsLeaf() const {
    if (m_mOutputs.empty())
      return true;

    for (const auto &clPair : m_mOutputs) {
      if (clPair.second != nullptr && clPair.second->HasTargets()) {
        const auto &vTargets = clPair.second->GetAllTargets();

        // Check for targets with outputs. We will ignore targets with no output.
        for (const auto &clTargetPair : vTargets) {
          std::shared_ptr<VertexType> p_clTarget = clTargetPair.first.lock();
          if (p_clTarget != nullptr && p_clTarget->HasOutputs())
            return false;
        }
      }
    }

    return true;
  }

  bool SetUseGPU(const bool &bUseGPU) {
    if (bUseGPU && !bleak::GetUseGPU()) {
      std::cerr << GetName() << ": Warning: Setting useGPU=true when GPU acceleration is globally disabled." << std::endl;
      return true;
    }
    m_bUseGPU = bUseGPU;
    return true;
  }

  bool GetUseGPU(bool &bUseGPU) const { 
    bUseGPU = m_bUseGPU; 
    return true;
  }

  virtual bool TestGradient() {
    if (GetAllOutputs().size() != 1) {
      std::cerr << GetName() << ": Info: Node does not have exactly 1 output. Skipping gradient test ..." << std::endl;
      return true;
    }

    const std::string &strOutputName = GetAllOutputs().begin()->first;

    return TestGradient(strOutputName);
  }

  // Use forward differencing to test gradients (this is computationally expensive)
  // Assumed SetSizes() and Initialize() has been called!
  virtual bool TestGradient(const std::string &strOutputName) {
    constexpr double dSmall = 1e-3;
    constexpr double dStep = 1e-2;

    bleakGetAndCheckOutput(p_clOutput, strOutputName, false);

    if (GetAllInputs().empty()) {
      std::cerr << GetName() << ": Info: Node does not have any inputs. Skipping gradient test ..." << std::endl;
      return true;
    }

    const ArrayType &clOutput = p_clOutput->GetData();
    ArrayType &clOutputGradient = p_clOutput->GetGradient();

    if (!clOutput.Valid()) {
      std::cout << GetName() << ": Error: No data initialized for output '" << strOutputName << "'." << std::endl;
      return false;
    }

    if (!clOutputGradient.Valid()) {
      std::cout << GetName() << ": Info: Node does not have output gradient. Skipping gradient test ..." << std::endl;
      return true;
    }

    const Size &clOutputSize = clOutput.GetSize();
    const int iOutputCount = clOutputSize.Count();

    if (clOutputSize[0] != 1) {
      std::cerr << GetName() << ": Error: Expected a batch size of 1 (got " << clOutputSize[0] << ")." << std::endl;
      return false;
    }

    // Initialize inputs
    std::normal_distribution<RealType> clNormalDist(RealType(0), RealType(1));

    bool bHasInputGradient = false;

    for (const auto &stPair : GetAllInputs()) {
      const std::string &strInputName = stPair.first;
      const std::shared_ptr<EdgeType> p_clInput = stPair.second.lock();

      if (!p_clInput) {
        std::cerr << GetName() << ": Warning: No input data set for '" << strInputName << "'." << std::endl;
        continue;
      }

      ArrayType &clInput = p_clInput->GetData();
      ArrayType &clInputGradient = p_clInput->GetGradient();

      if (!clInput.Valid()) {
        std::cerr << GetName() << ": Error No input data initialized for '" << strInputName << "'." << std::endl;
        return false;
      }

      if (!clInputGradient.Valid()) {
        std::cout << GetName() << ": Info: Node does not have input gradient for input '" << strInputName << "'. Skipping ..." << std::endl;
        continue;
      }

      bHasInputGradient = true;

      const Size &clInputSize = clInput.GetSize();
      const int iInputCount = clInputSize.Count();

      // XXX: Random input?
      std::generate_n(clInput.data_no_sync(), iInputCount, 
        [&clNormalDist]() -> RealType {
          return clNormalDist(GetGenerator());
        });
    }

    if (!bHasInputGradient) {
      std::cout << GetName() << ": Info: Node does not have any input gradients. Skipping gradient test ..." << std::endl;
      return true;
    }

    // Now run the tests...
    for (const auto &stPair : GetAllInputs()) {
      const std::string &strInputName = stPair.first;
      const std::shared_ptr<EdgeType> p_clInput = stPair.second.lock();

      if (!p_clInput || !p_clInput->GetGradient().Valid())
        continue;

      ArrayType &clInput = p_clInput->GetData();
      ArrayType &clInputGradient = p_clInput->GetGradient();

      const Size &clInputSize = clInput.GetSize();
      const int iInputCount = clInputSize.Count();

      std::cout << GetName() << ": Info: Testing input '" << strInputName << "' ..." << std::endl;

      ArrayType clOriginalInput; // Keep the original input (in case output/input memory is shared? ReLU for example?)
      ArrayType clOriginalGradient; // Centered at 0

      clOriginalInput.SetSize(clInputSize);
      clOriginalInput.Allocate();
      clOriginalGradient.SetSize(clInputSize);
      clOriginalGradient.Allocate();

      clInput.CopyTo(clOriginalInput);

      //const bool bSharedInputOutput = (clInput.data() == clOutput.data());
      //const bool bSharedInputOutputGradient = (clInputGradient.data() == clOutputGradient.data());

      // First, baseline
      Forward();
      clInputGradient.Fill(RealType()); // Do it in this order in case input is shared with output
      clOutputGradient.Fill(RealType(1));
      Backward();

      clInputGradient.CopyTo(clOriginalGradient);

      const RealType * const p_originalGradient = clOriginalGradient.data();

      for (int i = 0; i < iInputCount; ++i) {
        clOriginalInput.CopyTo(clInput);

        RealType *p_input = clInput.data();
        const RealType origValue = p_input[i]; // Keep a copy of the original

        p_input[i] = RealType(origValue + dStep);

        Forward();

        const RealType * p_output = clOutput.data(); // Copy it back to CPU

        double dTmpSumForward = 0.0;

#pragma omp parallel for reduction(+:dTmpSumForward)
        for (int k = 0; k < iOutputCount; ++k) {
          dTmpSumForward += p_output[k];
        }

        clOriginalInput.CopyTo(clInput); // Needed for shared memory reasons

        p_input = clInput.data();
        p_input[i] = RealType(origValue - dStep);

        Forward();

        p_output = clOutput.data(); // Copy it back to CPU

        double dTmpSumBackward = 0.0;

#pragma omp parallel for reduction(+:dTmpSumBackward)
        for (int k = 0; k < iOutputCount; ++k) {
          dTmpSumBackward += p_output[k];
        }

        const double dCenterDiff = 0.5*(dTmpSumForward - dTmpSumBackward)/dStep;
        const double dResidual = std::abs(dCenterDiff - p_originalGradient[i]);

        //if (dResidual > std::max(std::abs(dCenterDiff), std::abs((double)p_originalGradient[i]))*dSmall) {
        if (dResidual > dSmall) {
          std::cerr << GetName() << ": Error: Gradient may be incorrect with respect to input '" << strInputName << "': (x" << i << " = " << origValue << ", residual = " << dResidual << ", partial = " << p_originalGradient[i] << ")." << std::endl;
          //return false;
        }

      }
    }

    return true;
  }

protected:
  Vertex() = default;

  virtual void OnCreateWithNew() {
    m_bUseGPU = bleak::GetUseGPU();

    RegisterProperty("name", m_strName);
    RegisterProperty("saveOutputs", m_bSaveOutputs);
    RegisterGetterSetter("useGPU", std::bind(&Vertex::GetUseGPU, this, std::placeholders::_1), std::bind(&Vertex::SetUseGPU, this, std::placeholders::_1));
  }

  bool RegisterOutput(const std::string &strName) {
    if(strName.empty() || HasOutput(strName))
      return false;

    m_mOutputs.emplace(strName,std::make_shared<EdgeType>(shared_from_this()));

    return true;
  }

  bool RegisterInput(const std::string &strName) {
    return strName.size() > 0 ? m_mInputs.emplace(strName, std::weak_ptr<EdgeType>()).second : false;
  }

  template<typename ValueType>
  bool RegisterProperty(const std::string &strKey, ValueType &value) {
    if (strKey.empty() || m_mProperties.find(strKey) != m_mProperties.end())
      return false;

    m_mProperties.emplace(strKey, Property::New(value));

    return true;
  }

  template<typename GetterType, typename SetterType>
  bool RegisterGetterSetter(const std::string &strKey, GetterType funGetter, SetterType funSetter) {
    if (strKey.empty() || m_mProperties.find(strKey) != m_mProperties.end())
      return false;

    m_mProperties.emplace(strKey, Property::New(funGetter, funSetter));

    return true;
  }

  std::string MakeDatabaseKey(const std::string &strOutputName) const {
    return GetName() + '.' + strOutputName + ".data"; // In case we ever decide to store the gradient
  }

private:
  std::string m_strName;
  bool m_bSaveOutputs = false; // Whether or not to save this Vertex's output to the database (usually false).
  bool m_bUseGPU = false; // Whether or not to execute this vertex in GPU mode. Does nothing if GPU support is not compiled in.

  OutputMapType m_mOutputs;
  InputMapType m_mInputs;
  PropertyMapType m_mProperties;

  Vertex(const Vertex &) = delete;
  Vertex & operator=(const Vertex &) = delete;
};

} // end namespace bleak

#endif // !BLEAK_VERTEX_H
