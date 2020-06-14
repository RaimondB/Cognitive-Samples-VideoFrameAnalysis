#See https://aka.ms/containerfastmode to understand how Visual Studio uses this Dockerfile to build your images for faster debugging.

FROM mcr.microsoft.com/dotnet/core/runtime:3.1.5-bionic AS base
#FROM mcr.microsoft.com/dotnet/core/runtime:3.1-buster-slim AS base
WORKDIR /app
COPY ["CoreConsole/install-opencv-deps.sh", "/app"]
RUN /app/install-opencv-deps.sh

FROM mcr.microsoft.com/dotnet/core/sdk:3.1.301-bionic AS build
WORKDIR /src
COPY ["CoreConsole/CoreConsole.csproj", "CoreConsole/"]C:\Users\raimo\OneDrive\Repos\Cognitive-Samples-VideoFrameAnalysis\Windows\CoreConsole\Dockerfile
COPY ["VideoFrameAnalyzeStd/VideoFrameAnalyzeStd.csproj", "VideoFrameAnalyzeStd/"]
RUN dotnet restore "CoreConsole/CoreConsole.csproj"
COPY . .
WORKDIR "/src/CoreConsole"
RUN dotnet build "CoreConsole.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "CoreConsole.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "CoreConsole.dll"]